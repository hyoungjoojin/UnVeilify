import math

import torch
from torch import nn

from .generator import UNetStyleGAN
from .identity_extractor import IdentityExtractor
from .psp_encoder import PSPEncoder


class MaskRemover(nn.Module):
    def __init__(
        self,
        output_resolution: int = 512,
        latent_dim: int = 512,
        generator_arch: str = "UNetStyleGAN",
    ) -> None:
        super(MaskRemover, self).__init__()
        self.output_resolution = output_resolution
        self.log_resolution = int(math.log(output_resolution, 2))
        self.latent_dim = latent_dim

        # Number of styles for mapping to StyleGAN2 W+ space (ex. 18 for 1024x1024 output)
        self.n_styles = self.log_resolution * 2 - 2

        # Models
        self.psp_encoder = PSPEncoder(
            num_layers=50,
            mode="ir_se",
            n_styles=self.n_styles,
            input_resolution=output_resolution,
        )

        self.identity_extractor = IdentityExtractor(
            n_styles=self.n_styles, latent_dim=latent_dim
        )

        if generator_arch == "UNetStyleGAN":
            self.generator = UNetStyleGAN(self.output_resolution, 3, latent_dim)
        else:
            raise NotImplementedError

        # Parameters
        self.alpha = nn.Parameter(torch.tensor([0.5]))

        self.initialize_network()

    def forward(
        self,
        masked_image: torch.Tensor,
        identity_image: torch.Tensor,
    ):
        masked_image_features = self.psp_encoder(masked_image)
        identity_featues = self.identity_extractor(identity_image)

        latent_features = torch.add(masked_image_features, identity_featues)
        latent_features = (
            masked_image_features * self.alpha + identity_featues + (1 - self.alpha)
        )
        image = self.generator(masked_image, w=latent_features)
        return image

    def initialize_network(self):
        for m in self.generator.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data, 0.0)
