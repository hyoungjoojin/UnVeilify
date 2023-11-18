import math

import torch
from torch import nn

from .identity_extractor import IdentityExtractor
from .psp_encoder import PSPEncoder
from .stylegan2 import Generator


class MaskRemover(nn.Module):
    def __init__(self, output_resolution: int = 512, latent_dim: int = 512) -> None:
        super(MaskRemover, self).__init__()
        self.output_resolution = output_resolution
        self.log_resolution = int(math.log(output_resolution, 2))
        self.latent_dim = latent_dim

        # Number of styles for mapping to StyleGAN2 W+ space (ex. 18 for 1024x1024 output)
        self.n_styles = self.log_resolution * 2 - 2

        # Models
        self.psp_encoder = PSPEncoder(
            num_layers=50, mode="ir_se", n_styles=self.n_styles
        )
        self.identity_extractor = IdentityExtractor(
            n_styles=self.n_styles, latent_dim=latent_dim
        )
        self.generator = Generator(self.log_resolution, latent_dim)

        # Parameters
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(
        self,
        mask_image: torch.Tensor,
        identity_image: torch.Tensor,
    ):
        masked_image_features = self.psp_encoder(mask_image)
        identity_featues = self.identity_extractor(identity_image)

        latent_features = torch.add(masked_image_features, identity_featues)
        latent_features = (
            masked_image_features * self.alpha + identity_featues + (1 - self.alpha)
        )
        image = self.generator(w=latent_features)
        return image
