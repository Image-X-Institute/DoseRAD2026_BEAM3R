import torch


class Discriminator(torch.nn.Module):
    # based off a PatchGAN style network
    def __init__(
        self, in_channels: int, base_channels: int = 64, number_of_layers: int = 3
    ) -> None:
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.number_of_layers = number_of_layers

        self._initialise_architecture()

    def _initialise_architecture(self) -> None:
        layers = []
        layers.append(
            torch.nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.base_channels,
                kernel_size=(4, 4, 4),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
            )
        )
        layers.append(torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))

        curr_channels = self.base_channels
        for i in range(1, self.number_of_layers):
            next_channels = min(curr_channels * 2, 512)
            stride = 2 if i < self.number_of_layers - 1 else 1
            layers.append(
                torch.nn.Conv3d(
                    in_channels=curr_channels,
                    out_channels=next_channels,
                    kernel_size=(4, 4, 4),
                    stride=(stride, stride, stride),
                    padding=(1, 1, 1),
                )
            )
            layers.append(torch.nn.BatchNorm3d(next_channels))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
            curr_channels = next_channels
        layers.append(
            torch.nn.Conv3d(
                in_channels=curr_channels,
                out_channels=1,
                kernel_size=(4, 4, 4),
                stride=(stride, stride, stride),
                padding=(1, 1, 1),
            )
        )
        self.architecture = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.architecture(x)
