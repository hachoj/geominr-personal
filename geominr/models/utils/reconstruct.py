import os
import numpy as np
import SimpleITK as sitk
import torch
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SPACING = 0.0335860058309
PROBE_RADIUS = 12.5

# Pad the per-batch query count to a fixed power-of-two bucket so torch.compile
# sees a small, bounded set of query shapes instead of a new one per batch (the
# data-dependent Nmax would otherwise trigger a recompile every batch).
PAD_NMAX_TO_BUCKET = True


from .utils import trilinear_sample


def _read_one_slice(args):
    slice_path, filepath = args
    # A fresh reader per call: sitk.ImageFileReader is stateful (SetFileName
    # mutates it), so a shared reader is NOT thread-safe.
    reader = sitk.ImageFileReader()
    reader.SetFileName(filepath)
    reader.ReadImageInformation()

    img = reader.Execute()

    vol = sitk.GetArrayFromImage(img)  # (1, H, W, C?) or (H, W) depending on input
    if vol.ndim == 4:
        # intensity stored as R=G=B use R
        vol_2d = vol[0, :, :, 0]
    elif vol.ndim == 3:
        # (1, H, W)
        vol_2d = vol[0, :, :]
    else:
        # already (H, W)
        vol_2d = vol

    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    size = img.GetSize()  # (W, H, [C])

    angle = origin[2]

    return slice_path, {
        "angle": float(angle),
        "slice_volume": vol_2d.astype(np.uint8, copy=False),
        "shape": size,  # (W, H, [C])
        "spacing": spacing,  # (sx, sy, [sz])
    }


def extract_slices(dicom_patient_dir):
    patient_slices_paths = sorted(os.listdir(dicom_patient_dir))

    # Read DICOMs in parallel: SimpleITK releases the GIL during file I/O, and
    # each read is independent (its own reader). Insertion order is irrelevant —
    # every downstream consumer re-sorts the dict by key.
    args = [
        (sp, os.path.join(dicom_patient_dir, sp)) for sp in patient_slices_paths
    ]
    max_workers = min(8, len(args)) if args else 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        slices = dict(ex.map(_read_one_slice, args))

    return slices


def save_volume(extracted_slices, reconstructed, patient_dir):
    os.makedirs(os.path.dirname(patient_dir), exist_ok=True)

    vol = reconstructed["volume"]  # (LR, AP, SI)
    spacing_LR = reconstructed["spacing_LR"]
    spacing_AP = reconstructed["spacing_AP"]
    spacing_SI = reconstructed["spacing_SI"]

    itk_img = sitk.GetImageFromArray(vol.transpose(2, 1, 0))

    itk_img.SetOrigin((0.0, 0.0, 0.0))
    # Spacing order in SimpleITK is (x, y, z) == (LR, AP, SI)
    itk_img.SetSpacing((float(spacing_LR), float(spacing_AP), float(spacing_SI)))
    itk_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    out_path = f"{patient_dir}.nii.gz"
    sitk.WriteImage(itk_img, out_path)
    print(f"Saved: {out_path}")


def _prepare_slices(extracted_slices, device, sample_dtype):
    items = sorted(extracted_slices.items(), key=lambda kv: kv[0])
    angles_arr = np.asarray([v["angle"] for _, v in items], dtype=float)  # (S,)
    slices_list = [v["slice_volume"] for _, v in items]

    H = int(slices_list[0].shape[0])
    W = int(slices_list[0].shape[1])
    slices_stack = np.stack(slices_list, axis=0)  # (S,H,W)

    sort_idx = np.argsort(angles_arr)
    angles_sorted = torch.from_numpy(angles_arr[sort_idx]).to(
        device=device, dtype=torch.float32
    )  # (S,)
    slices_sorted = slices_stack[sort_idx, :, :]

    slices_t = torch.from_numpy(slices_sorted).to(
        device=device, dtype=sample_dtype
    )  # (S,H,W)

    spacing_xy = float(items[0][1]["spacing"][0])
    return angles_sorted, slices_t, H, W, spacing_xy


def _prepare_geometry(
    *,
    depth: int,
    ap_voxel_count: int,
    probe_radius: float,
    spacing_xy: float,
    H: int,
    W: int,
    device: torch.device,
    geom_dtype: torch.dtype,
):
    height_mm = spacing_xy * H
    width_mm = spacing_xy * W

    AP_mm = height_mm + probe_radius
    LR_mm = 2.0 * AP_mm
    SI_mm = width_mm

    AP_num = int(ap_voxel_count)
    LR_num = int(2 * ap_voxel_count)
    SI_num = int(depth)
    if AP_num <= 0 or LR_num <= 0 or SI_num <= 0:
        raise ValueError("Derived volume dimensions must be positive")

    spacing_AP = AP_mm / float(AP_num)
    spacing_LR = LR_mm / float(LR_num)
    spacing_SI = SI_mm / float(SI_num)

    xs = torch.arange(LR_num, device=device, dtype=geom_dtype) * spacing_LR
    ys = torch.arange(AP_num, device=device, dtype=geom_dtype) * spacing_AP
    zs = torch.arange(SI_num, device=device, dtype=geom_dtype) * spacing_SI

    Xmm = xs[:, None, None]  # (LR,1,1)
    Ymm = ys[None, :, None]  # (1,AP,1)
    Zmm = zs[None, None, :]  # (1,1,SI)

    height_plus_r = float(height_mm + probe_radius)
    inv_spacing_xy = float(1.0 / spacing_xy)

    dist_mid = Xmm - height_plus_r
    dist_post = height_plus_r - Ymm

    theta = -torch.rad2deg(
        torch.atan(dist_mid / torch.clamp(dist_post, min=1e-8))
    )  # (LR,AP,SI)
    d = torch.sqrt(dist_mid**2 + dist_post**2)

    j_idx = ((height_plus_r - d) * inv_spacing_xy).to(torch.int32)
    i_idx = (Zmm * inv_spacing_xy).to(torch.int32)

    return (
        theta,
        j_idx,
        i_idx,
        LR_num,
        AP_num,
        SI_num,
        spacing_LR,
        spacing_AP,
        spacing_SI,
    )


def reconstruct_volume_nn(
    extracted_slices,
    depth,
    ap_voxel_count,
    probe_radius=12.5,
    *,
    device=None,
    geom_dtype=torch.float32,
    sample_dtype=torch.uint8,
):
    if depth <= 0:
        raise ValueError("depth must be a positive integer")
    if ap_voxel_count <= 0:
        raise ValueError("ap_voxel_count must be a positive integer")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    angles_sorted, slices_t, H, W, spacing_xy = _prepare_slices(
        extracted_slices, device, sample_dtype
    )

    (
        theta,
        j_idx,
        i_idx,
        LR_num,
        AP_num,
        SI_num,
        spacing_LR,
        spacing_AP,
        spacing_SI,
    ) = _prepare_geometry(
        depth=depth,
        ap_voxel_count=ap_voxel_count,
        probe_radius=probe_radius,
        spacing_xy=spacing_xy,
        H=H,
        W=W,
        device=device,
        geom_dtype=geom_dtype,
    )

    angle_min = float(angles_sorted.min().item())
    angle_max = float(angles_sorted.max().item())
    covered = (theta >= angle_min) & (theta <= angle_max)
    valid_ij = (j_idx >= 0) & (j_idx < H) & (i_idx >= 0) & (i_idx < W)
    valid = covered & valid_ij

    angles_sorted_f = angles_sorted.to(theta.dtype)
    idx = torch.bucketize(theta, angles_sorted_f)  # 0..S
    S = angles_sorted.shape[0]
    idx0 = (idx - 1).clamp(0, S - 1)
    idx1 = idx.clamp(0, S - 1)

    a0 = angles_sorted_f[idx0]
    a1 = angles_sorted_f[idx1]
    choose_idx1 = (theta - a0).abs() > (theta - a1).abs()
    angle_idx = torch.where(choose_idx1, idx1, idx0).to(torch.int64)

    linear_idx = (j_idx.to(torch.int64) * W + i_idx.to(torch.int64)).clamp(min=0)
    angle_idx = torch.where(valid, angle_idx, torch.zeros_like(angle_idx))
    linear_idx = torch.where(valid, linear_idx, torch.zeros_like(linear_idx))

    slices_flat = slices_t.reshape(S, H * W)
    out = slices_flat[angle_idx.reshape(-1), linear_idx.reshape(-1)]
    out = out.reshape(LR_num, AP_num, SI_num)
    out = torch.where(valid, out, torch.zeros_like(out))

    result = {
        "volume": out.to(torch.uint8).cpu().numpy(),
        "spacing_LR": spacing_LR,
        "spacing_AP": spacing_AP,
        "spacing_SI": spacing_SI,
    }
    return result


def reconstruct_volume_sr_fast(
    model,
    extracted_slices,
    depth,
    ap_voxel_count,
    patch_size,
    stride,
    probe_radius=12.5,
    *,
    max_queries_per_batch=100_000_000,
    tiles_per_batch=1024,
    geom_dtype=torch.float32,
    infer_dtype=torch.bfloat16,
    show_progress=True,
):
    if depth <= 0:
        raise ValueError("depth must be a positive integer")
    if ap_voxel_count <= 0:
        raise ValueError("ap_voxel_count must be a positive integer")

    items = sorted(extracted_slices.items(), key=lambda kv: kv[0])
    angles_arr = np.asarray([v["angle"] for _, v in items], dtype=float)  # (S,)
    slices_list = [v["slice_volume"] for _, v in items]
    H = int(slices_list[0].shape[0])
    W = int(slices_list[0].shape[1])
    slices_stack = np.stack(slices_list, axis=0)  # (S,H,W)

    sort_idx = np.argsort(angles_arr)
    angles_sorted = angles_arr[sort_idx]  # (S,)
    slices_sorted = slices_stack[sort_idx, :, :]  # (S,H,W)

    spacing_xy = float(items[0][1]["spacing"][0])
    height_mm = spacing_xy * H
    width_mm = spacing_xy * W

    AP_mm = height_mm + probe_radius
    LR_mm = 2.0 * AP_mm
    SI_mm = width_mm

    AP_num = int(ap_voxel_count)
    LR_num = int(2 * ap_voxel_count)
    SI_num = int(depth)
    if AP_num <= 0 or LR_num <= 0 or SI_num <= 0:
        raise ValueError("Derived volume dimensions must be positive")

    spacing_AP = AP_mm / float(AP_num)
    spacing_LR = LR_mm / float(LR_num)
    spacing_SI = SI_mm / float(SI_num)

    device = next(model.parameters()).device
    prev_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    slices_t = torch.from_numpy(slices_sorted).to(
        device=device, dtype=torch.float32
    )  # [S,H,W]
    slices_t = slices_t / 255.0  # [0,1]

    height_plus_r = float(height_mm + probe_radius)
    inv_spacing_xy = float(1.0 / spacing_xy)

    xs = torch.arange(LR_num, device=device, dtype=geom_dtype) * spacing_LR  # LR
    ys = torch.arange(AP_num, device=device, dtype=geom_dtype) * spacing_AP  # AP
    zs = torch.arange(SI_num, device=device, dtype=geom_dtype) * spacing_SI  # SI

    Xmm = xs[:, None, None]  # (LR,1,1)
    Ymm = ys[None, :, None]  # (1,AP,1)
    Zmm = zs[None, None, :]  # (1,1,SI)

    dist_mid = Xmm - height_plus_r
    dist_post = height_plus_r - Ymm

    theta = -torch.rad2deg(
        torch.atan(dist_mid / torch.clamp(dist_post, min=1e-8))
    )  # (LR,AP,SI)
    d = torch.sqrt(dist_mid**2 + dist_post**2)

    j_idx = ((height_plus_r - d) * inv_spacing_xy).to(torch.int32)
    i_idx = (Zmm * inv_spacing_xy).to(torch.int32)

    # The geometry factorizes: theta and j_idx depend only on the (LR,AP) plane
    # (Xmm, Ymm); i_idx depends only on SI (Zmm). Keep them as 2D/1D tensors
    # rather than expanding to the full [LR,AP,SI] volume -- voxel membership in
    # a tile is then a plane test AND an SI test, combined as a Cartesian product
    # in the job loop below (replacing the full-volume boolean scan + nonzero).
    theta2 = theta.reshape(LR_num, AP_num)  # [LR,AP]
    j2 = j_idx.reshape(LR_num, AP_num)  # [LR,AP]
    i1 = i_idx.reshape(SI_num)  # [SI]

    angle_min = float(angles_sorted.min())
    angle_max = float(angles_sorted.max())

    # Plane voxels that are angularly covered with an in-range radial index; SI
    # indices with an in-range column index. The full validity mask is their
    # outer product (valid[l,a,s] = plane_valid[l,a] & ivalid1[s]).
    plane_valid = (
        (theta2 >= angle_min) & (theta2 <= angle_max) & (j2 >= 0) & (j2 < H)
    )  # [LR,AP]
    ivalid1 = (i1 >= 0) & (i1 < W)  # [SI]

    total_valid = int(plane_valid.sum().item()) * int(ivalid1.sum().item())
    wedge_total = max(0, len(angles_sorted) - 1)
    vox_bar = tqdm(
        total=total_valid,
        unit="vox",
        unit_scale=True,
        desc="Voxels",
        disable=not show_progress,
        position=0,
    )
    wedge_bar = tqdm(
        total=wedge_total,
        desc="Wedges",
        disable=not show_progress,
        position=1,
        leave=False,
    )

    Ph, Pw = patch_size
    Sh, Sw = stride

    def tile_starts(full, size, step):
        starts = list(range(0, max(1, full - size + 1), step))
        if len(starts) == 0 or starts[-1] != full - size:
            starts.append(max(0, full - size))
        return starts

    row_starts = tile_starts(H, Ph, Sh)
    col_starts = tile_starts(W, Pw, Sw)

    total_vox = LR_num * AP_num * SI_num
    acc_sum_flat = torch.zeros(total_vox, device=device, dtype=torch.float32)
    acc_wts_flat = torch.zeros_like(acc_sum_flat)

    def run_batch(jobs):
        if not jobs:
            return

        Ns = [j["N"] for j in jobs]
        nmax_raw = int(max(Ns))
        # Pad the query count to a fixed power-of-two bucket (floored at 64) so
        # torch.compile sees a small, bounded set of query shapes instead of a new
        # one per batch (the data-dependent Nmax otherwise triggers a recompile
        # every batch). The decoder is per-query and its cost is independent of
        # Nmax, so the extra rows are nearly free; they are zero-coord padding that
        # is discarded by the pred[b, :n] unpad below and never scattered.
        if PAD_NMAX_TO_BUCKET:
            Nmax = max(64, 1 << (nmax_raw - 1).bit_length()) if nmax_raw > 0 else 64
        else:
            Nmax = nmax_raw
        B = len(jobs)

        coords_b = torch.zeros((B, Nmax, 3), device=device, dtype=torch.float32)
        cond_b = torch.empty((B, 2, Ph, Pw), device=device, dtype=torch.float32)
        arcs_b = torch.empty((B, Ph), device=device, dtype=torch.float32)

        flat_idx_list = []
        valid_counts = []

        for b, j in enumerate(jobs):
            n = j["N"]
            coords_b[b, :n, 0] = j["x_norm"]
            coords_b[b, :n, 1] = j["y_norm"]
            coords_b[b, :n, 2] = j["z_norm"]

            cond_b[b, 0] = slices_t[j["k"], j["r0"] : j["r1"], j["c0"] : j["c1"]]
            cond_b[b, 1] = slices_t[j["k"] + 1, j["r0"] : j["r1"], j["c0"] : j["c1"]]

            rows = torch.arange(j["r0"], j["r1"], device=device, dtype=torch.float32)
            delta_angle_rad = torch.abs(j["a0"] - j["a1"]) * (torch.pi / 180.0)
            arcs_b[b] = (rows * SPACING + PROBE_RADIUS) * delta_angle_rad

            flat_idx_list.append(j["flat_idx"])
            valid_counts.append(n)

        flat_idx = torch.cat(flat_idx_list, dim=0)

        left_slice = cond_b[:, 0].unsqueeze(1)
        right_slice = cond_b[:, 1].unsqueeze(1)

        with torch.amp.autocast(
            device_type=device.type, dtype=infer_dtype
        ), torch.no_grad():
            pred = model(left_slice, right_slice, arcs_b, coords_b)

        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)

        outs = []
        for b, n in enumerate(valid_counts):
            outs.append(pred[b, :n].float())
        p = torch.cat(outs, dim=0)

        p01 = torch.clamp(p, 0.0, 1.0)
        p255 = p01 * 255.0

        acc_sum_flat.scatter_add_(0, flat_idx, p255)
        acc_wts_flat.scatter_add_(
            0, flat_idx, torch.ones_like(p255, dtype=torch.float32)
        )

        if show_progress:
            vox_bar.update(int(p255.numel()))

    # Per col-tile SI membership (independent of wedge/row): precompute once.
    # Each entry is (si_indices [ascending int64], x_norm) for that column tile.
    APSI = AP_num * SI_num
    si_by_col = []
    for c0 in col_starts:
        c1 = c0 + Pw
        col_ok = ivalid1 & (i1 >= c0) & (i1 < c1)  # [SI]
        si_c = torch.nonzero(col_ok, as_tuple=True)[0]  # ascending si
        i_local_c = (i1[si_c] - c0).to(torch.int32)
        x_norm_c = ((i_local_c.to(geom_dtype) + 0.5) / Pw) * 2.0 - 1.0
        si_by_col.append((si_c.to(torch.int64), x_norm_c))

    pending = []
    pending_queries = 0

    for k in range(len(angles_sorted) - 1):
        a0 = float(angles_sorted[k])
        a1 = float(angles_sorted[k + 1])
        a0_t = torch.tensor(a0, device=device, dtype=torch.float32)
        a1_t = torch.tensor(a1, device=device, dtype=torch.float32)
        da = max(a1 - a0, 1e-6)
        # Plane voxels in this wedge. Closed interval on both ends, matching the
        # original wmask: a voxel with theta == a1 is emitted in BOTH this wedge
        # and the next, preserving the overlap multiplicity.
        wk_plane = plane_valid & (theta2 >= a0) & (theta2 <= a1)  # [LR,AP]

        for r0 in row_starts:
            r1 = r0 + Ph
            P = wk_plane & (j2 >= r0) & (j2 < r1)  # [LR,AP]
            lr_p, ap_p = torch.nonzero(P, as_tuple=True)  # (lr,ap)-lex order
            Np = lr_p.numel()
            if Np == 0:
                continue

            jj_p = j2[lr_p, ap_p]  # [Np]
            th_p = theta2[lr_p, ap_p]  # [Np]
            j_local_p = (jj_p - r0).to(torch.int32)
            y_norm_p = ((j_local_p.to(geom_dtype) + 0.5) / Ph) * 2.0 - 1.0  # [Np]
            z_p = (torch.clamp((th_p - a0) / da, 0.0, 1.0) - 0.5).to(geom_dtype)  # [Np]
            plane_base = lr_p.to(torch.int64) * APSI + ap_p.to(torch.int64) * SI_num

            for ci, c0 in enumerate(col_starts):
                si_c, x_norm_c = si_by_col[ci]
                Nq = si_c.numel()
                if Nq == 0:
                    continue

                # Cartesian product of the Np plane voxels with the Nq SI indices,
                # emitted in (lr, ap, si) lexicographic order (plane-major,
                # si-minor) -- identical to the original torch.nonzero(in_tile)
                # ordering, so the downstream scatter result is unchanged.
                flat_idx = (plane_base[:, None] + si_c[None, :]).reshape(-1)
                x_norm = x_norm_c[None, :].expand(Np, Nq).reshape(-1)
                y_norm = y_norm_p[:, None].expand(Np, Nq).reshape(-1)
                z_norm = z_p[:, None].expand(Np, Nq).reshape(-1)
                N = Np * Nq

                job = {
                    "k": k,
                    "a0": a0_t,
                    "a1": a1_t,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c0 + Pw,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "z_norm": z_norm,
                    "flat_idx": flat_idx,
                    "N": int(N),
                }

                pending.append(job)
                pending_queries += N

                if (pending_queries >= max_queries_per_batch) or (
                    len(pending) >= tiles_per_batch
                ):
                    run_batch(pending)
                    pending.clear()
                    pending_queries = 0

        if show_progress:
            wedge_bar.update(1)

    if pending:
        run_batch(pending)

    covered_mask = acc_wts_flat > 0
    vol_flat = torch.zeros_like(acc_sum_flat, dtype=torch.uint8)
    avg = (
        (acc_sum_flat[covered_mask] / acc_wts_flat[covered_mask]).round().clamp(0, 255)
    )
    vol_flat[covered_mask] = avg.to(torch.uint8)
    vol = vol_flat.view(LR_num, AP_num, SI_num).cpu().numpy()

    if show_progress:
        vox_bar.close()
        wedge_bar.close()

    result = {
        "volume": vol,  # (LR, AP, SI)
        "spacing_LR": spacing_LR,
        "spacing_AP": spacing_AP,
        "spacing_SI": spacing_SI,
    }
    torch.set_grad_enabled(prev_grad_enabled)
    return result


def reconstruct_volume_linear_fast(
    extracted_slices,
    depth,
    ap_voxel_count,
    probe_radius=12.5,
    *,
    device=None,
    geom_dtype=torch.float32,
    sample_dtype=torch.uint8,
):
    if depth <= 0:
        raise ValueError("depth must be a positive integer")
    if ap_voxel_count <= 0:
        raise ValueError("ap_voxel_count must be a positive integer")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    angles_sorted, slices_t, H, W, spacing_xy = _prepare_slices(
        extracted_slices, device, sample_dtype
    )

    (
        theta,
        j_idx,
        i_idx,
        LR_num,
        AP_num,
        SI_num,
        spacing_LR,
        spacing_AP,
        spacing_SI,
    ) = _prepare_geometry(
        depth=depth,
        ap_voxel_count=ap_voxel_count,
        probe_radius=probe_radius,
        spacing_xy=spacing_xy,
        H=H,
        W=W,
        device=device,
        geom_dtype=geom_dtype,
    )

    angle_min = float(angles_sorted.min().item())
    angle_max = float(angles_sorted.max().item())
    covered = (theta >= angle_min) & (theta <= angle_max)
    valid_ij = (j_idx >= 0) & (j_idx < H) & (i_idx >= 0) & (i_idx < W)
    valid = covered & valid_ij

    angles_sorted_f = angles_sorted.to(theta.dtype)
    idx = torch.bucketize(theta, angles_sorted_f)
    S = angles_sorted.shape[0]
    idx1 = idx.clamp(0, S - 1)
    idx0 = (idx1 - 1).clamp(0, S - 1)

    a0 = angles_sorted_f[idx0]
    a1 = angles_sorted_f[idx1]
    denom = torch.clamp(a1 - a0, min=1e-6)
    w = torch.clamp((theta - a0) / denom, 0.0, 1.0)
    w = w.expand_as(valid).contiguous()

    linear_idx = (j_idx.to(torch.int64) * W + i_idx.to(torch.int64)).clamp(min=0)
    idx0 = torch.where(valid, idx0, torch.zeros_like(idx0))
    idx1 = torch.where(valid, idx1, torch.zeros_like(idx1))
    linear_idx = torch.where(valid, linear_idx, torch.zeros_like(linear_idx))

    slices_flat = slices_t.reshape(S, H * W)
    left = slices_flat[idx0.reshape(-1), linear_idx.reshape(-1)].to(torch.float32)
    right = slices_flat[idx1.reshape(-1), linear_idx.reshape(-1)].to(torch.float32)

    w_flat = w.reshape(-1).to(torch.float32)
    out = left + (right - left) * w_flat
    out = out.reshape(LR_num, AP_num, SI_num)
    out = torch.where(valid, out, torch.zeros_like(out))

    result = {
        "volume": out.round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
        "spacing_LR": spacing_LR,
        "spacing_AP": spacing_AP,
        "spacing_SI": spacing_SI,
    }
    return result
