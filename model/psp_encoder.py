import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utils import bottleneck_IR, bottleneck_IR_SE, get_blocks


class PSPEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_layers: int = 50,
        mode="ir",
        n_styles: int = 16,
        input_resolution: int = 512,
    ):
        super(PSPEncoder, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)

        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        else:
            raise NotImplementedError

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, input_resolution // 2**5)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, input_resolution // 2**4)
            else:
                style = GradualStyleBlock(512, 512, input_resolution // 2**3)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.initialize_network()

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)
        print(x.shape)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            print(x.shape)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out

    def initialize_network(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data, 0.0)


class GradualStyleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, resolution: int):
        super(GradualStyleBlock, self).__init__()
        self.out_channels = out_channels

        modules = []
        for _ in range(int(np.log2(resolution))):
            modules += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.LeakyReLU(),
            ]
        modules.append(nn.MaxPool2d(kernel_size=2))

        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(in_features=out_channels, out_features=out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convs(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x
