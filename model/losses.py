from typing import Dict, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

__all__ = ["MaskRemoverLoss"]


def mse_loss(pred, target, normalize: bool = True):
    if normalize:
        pred = F.normalize(pred)
        target = F.normalize(target)
    return F.mse_loss(pred, target, reduction="mean")


class MaskRemoverLoss(nn.Module):
    def __init__(
        self,
        lambda_content: float,
        lambda_perceptual: float,
        lambda_identity: float,
        lambda_adversarial: float,
    ) -> None:
        super(MaskRemoverLoss, self).__init__()
        self.lambda_content = lambda_content
        self.lambda_perceptual = lambda_perceptual
        self.lambda_identity = lambda_identity
        self.lambda_adversarial = lambda_adversarial

        self.perceptual_loss = lpips.LPIPS(net="vgg", spatial=False)
        self.identity_loss = IdentityLoss()
        self.gan_loss = F.binary_cross_entropy

    def forward(
        self,
        ground_truth: torch.Tensor,
        generated_image: torch.Tensor,
        identity_image: torch.Tensor,
        discriminator: nn.Module,
        optim_d: torch.optim.Optimizer,
    ) -> Tuple[torch.Tensor, Dict]:
        output_real = discriminator(ground_truth, identity_image).view(-1)
        output_fake = discriminator(generated_image, identity_image).view(-1)

        loss_real = self.gan_loss(output_real, torch.ones_like(output_real))
        loss_fake = self.gan_loss(output_fake, torch.zeros_like(output_fake))
        loss_d = (loss_real + loss_fake) / 2

        optim_d.zero_grad()
        loss_d.backward(retain_graph=True)
        optim_d.step()

        output_fake = discriminator(generated_image, identity_image).view(-1)
        adverarial_loss = self.gan_loss(output_fake, torch.ones_like(output_fake))

        content_loss = mse_loss(ground_truth, generated_image)
        identity_loss = self.identity_loss(ground_truth, generated_image)
        perceptual_loss = self.perceptual_loss(ground_truth, generated_image).mean()

        loss = (
            self.lambda_content * content_loss
            + self.lambda_identity * identity_loss
            + self.lambda_perceptual * perceptual_loss
            + self.lambda_adversarial * adverarial_loss
        )

        return loss, {
            "content": content_loss.item(),
            "perceptual": perceptual_loss.item(),
            "identity": identity_loss.item(),
            "adversarial": adverarial_loss.item(),
        }


class IdentityLoss(nn.Module):
    def __init__(self) -> None:
        super(IdentityLoss, self).__init__()

        self.facenet = InceptionResnetV1(pretrained="vggface2").eval()
        self.activation = {}
        self.facenet.dropout.register_forward_hook(
            lambda _model, _input, output: self.activation.__setitem__(
                "dropout", output
            )
        )

    def extract_feats(self, x: torch.Tensor):
        with torch.no_grad():
            self.facenet(x)

        x_feats = self.activation["dropout"].squeeze()
        return x_feats

    def forward(
        self, ground_truth: torch.Tensor, generated_image: torch.Tensor
    ) -> torch.Tensor:
        gt_feats = self.extract_feats(ground_truth)
        pred_feats = self.extract_feats(generated_image)
        return mse_loss(gt_feats, pred_feats, normalize=False)
