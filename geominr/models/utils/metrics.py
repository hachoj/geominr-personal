import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

def PSNR(
    x: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
    max_value: float = 1.0,
    eps: float = 1e-8,
):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match.")
    x = x.float()
    y = y.float()
    diff = (x - y) ** 2
    if diff.dim() == 0:
        mse = diff
    else:
        reduce_dims = tuple(range(1, diff.dim()))
        mse = diff.mean(dim=reduce_dims)
    mse = mse.clamp_min(eps)
    max_tensor = torch.tensor(max_value, dtype=x.dtype, device=x.device)
    psnr = 10.0 * torch.log10((max_tensor**2) / mse)
    if reduction == "mean":
        return psnr.mean()
    if reduction == "sum":
        return psnr.sum()
    if reduction == "none":
        return psnr
    raise ValueError(f"Unsupported reduction: {reduction}")


def SSIM_slicewise(x, y, reduction="mean"):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match.")
    x = x.float()
    y = y.float()
    data_range = 1.0
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range, reduction="none").to(
        x.device
    )
    values = ssim(x, y)
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction: {reduction}")


def SSI(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 32,
    reduction: str = "mean",
    eps: float = 1e-8,
):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match.")

    B, C, H, W = x.shape
    x = x.float()
    y = y.float()

    unfold = torch.nn.Unfold(kernel_size=window_size, stride=window_size)
    x_patches = unfold(x).view(B, C, -1, window_size * window_size)
    y_patches = unfold(y).view(B, C, -1, window_size * window_size)

    mu_x = x_patches.mean(dim=-1)
    mu_y = y_patches.mean(dim=-1)
    sigma_x = x_patches.std(dim=-1, unbiased=False) + eps
    sigma_y = y_patches.std(dim=-1, unbiased=False) + eps

    num = 2 * sigma_x * sigma_y + eps
    den = sigma_x**2 + sigma_y**2 + (mu_x - mu_y) ** 2 + eps
    ssi_local = num / den
    ssi = ssi_local.mean(dim=-1)

    if reduction == "mean":
        return ssi.mean()
    if reduction == "sum":
        return ssi.sum()
    if reduction == "none":
        return ssi
    raise ValueError(f"Unsupported reduction: {reduction}")


def EPI(
    x: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match.")
    x = x.float()
    y = y.float()

    sobel_x = (
        torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sobel_y = (
        torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    gx_x = F.conv2d(x, sobel_x, padding=1)
    gy_x = F.conv2d(x, sobel_y, padding=1)
    gx_y = F.conv2d(y, sobel_x, padding=1)
    gy_y = F.conv2d(y, sobel_y, padding=1)

    mag_x = torch.sqrt(gx_x**2 + gy_x**2 + eps)
    mag_y = torch.sqrt(gx_y**2 + gy_y**2 + eps)

    mu_x = mag_x.mean(dim=[1, 2, 3])
    mu_y = mag_y.mean(dim=[1, 2, 3])

    num = ((mag_x - mu_x.view(-1, 1, 1, 1)) * (mag_y - mu_y.view(-1, 1, 1, 1))).sum(
        dim=[1, 2, 3]
    )
    den = torch.sqrt(
        ((mag_x - mu_x.view(-1, 1, 1, 1)) ** 2).sum(dim=[1, 2, 3])
        * ((mag_y - mu_y.view(-1, 1, 1, 1)) ** 2).sum(dim=[1, 2, 3])
        + eps
    )

    epi = num / den

    if reduction == "mean":
        return epi.mean()
    if reduction == "sum":
        return epi.sum()
    if reduction == "none":
        return epi
    raise ValueError(f"Unsupported reduction: {reduction}")
