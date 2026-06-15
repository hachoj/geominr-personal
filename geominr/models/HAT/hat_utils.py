import torch
from torch import Tensor


def create_local_window(x: Tensor, M: int) -> Tensor:
    """
    Args:
        x: [B,H,W,C]
        M: window size
    Returns:
        x: [B,num_windows,M*M,C]
    """
    B, H, W, C = x.shape
    h_chunks = H // M
    w_chunks = W // M

    x = x.view(B, h_chunks, M, w_chunks, M, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B,h_chunks,w_chunks,M,M,C]
    x = x.reshape(B, h_chunks * w_chunks, M * M, C)

    return x


def reverse_local_window(x: Tensor, M: int, H: int, W: int) -> Tensor:
    """
    Args:
        x: [B,num_windows,M*M,C]
        M: window size
        H: original height
        W: original width
    Returns:
        x: [B,H,W,C]
    """
    B, num_windows, _, C = x.shape
    h_chunks = H // M
    w_chunks = W // M

    x = x.view(B, h_chunks, w_chunks, M, M, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B,h_chunks,M,w_chunks,M,C]
    x = x.reshape(B, H, W, C)

    return x
