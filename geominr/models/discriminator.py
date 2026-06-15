import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, base_channels, kernel_size=4, stride=2, padding=1
            ),  # H/2 x W/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,  # H/4 x W/4
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,  # H/8 x W/8
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
        )
        self.out = nn.Conv2d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: Tensor, [B,1,H,W]
        Returns:
            out: Tensor, [B,1,H/8,W/8]
        """
        x = self.net(x)
        return self.out(x)
