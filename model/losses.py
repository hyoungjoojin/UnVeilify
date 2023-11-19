import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

__all__ = ["MaskRemoverLoss"]


def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="mean")


class MaskRemoverLoss(nn.Module):
    def __init__(
        self, lambda_content: float, lambda_perceptual: float, lambda_identity: float
    ) -> None:
        super(MaskRemoverLoss, self).__init__()
        self.lambda_content = lambda_content
        self.lambda_perceptual = lambda_perceptual
        self.lambda_identity = lambda_identity

        self.perceptual_loss = lpips.LPIPS(net="vgg", spatial=False)
        self.identity_loss = IdentityLoss()

    def forward(
        self, ground_truth: torch.Tensor, generated_image: torch.Tensor
    ) -> torch.Tensor:
        content_loss = mse_loss(ground_truth, generated_image)
        identity_loss = self.identity_loss(ground_truth, generated_image)
        perceptual_loss = self.perceptual_loss(ground_truth, generated_image).mean()

        loss = (
            self.lambda_content * content_loss
            + self.lambda_identity * identity_loss
            + self.lambda_perceptual * perceptual_loss
        )

        return loss


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
        return mse_loss(gt_feats, pred_feats)


# TODO: Why does PSP not use GAN loss?
class GANLoss(nn.Module):
    def __init__(self) -> None:
        super(GANLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
