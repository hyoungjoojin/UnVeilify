from typing import Dict, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bottleneck_IR, bottleneck_IR_SE, get_blocks, l2_norm

__all__ = ["MaskRemoverLoss"]


def mse_loss(pred, target, normalize: bool = False):
    if normalize:
        pred = F.normalize(pred)
        target = F.normalize(target)
    return F.mse_loss(pred, target, reduction="sum")


class MaskRemoverLoss(nn.Module):
    def __init__(
        self,
        pretrained_irse50: str,
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
        self.identity_loss = IdentityLoss(pretrained_irse50)
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
        perceptual_loss = self.perceptual_loss(ground_truth, generated_image).sum()

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
    def __init__(self, pretrained_irse50: str):
        super(IdentityLoss, self).__init__()
        print("Loading ResNet ArcFace")
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(torch.load(pretrained_irse50))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, generated_image, ground_truth):
        batch_size = generated_image.shape[0]
        ground_truth_feats = self.extract_feats(ground_truth).detach()
        generated_image_feats = self.extract_feats(generated_image)

        loss = 0

        for i in range(batch_size):
            diff_target = generated_image_feats[i].dot(ground_truth_feats[i])
            loss += 1 - diff_target

        return loss


class Backbone(nn.Module):
    def __init__(self, input_size, num_layers, mode="ir", drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        if input_size == 112:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(drop_ratio),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 512),
                nn.BatchNorm1d(512, affine=affine),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(drop_ratio),
                nn.Flatten(),
                nn.Linear(512 * 14 * 14, 512),
                nn.BatchNorm1d(512, affine=affine),
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

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)
