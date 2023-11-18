import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class IdentityExtractor(nn.Module):
    def __init__(self, n_styles: int, latent_dim: int = 256) -> None:
        super(IdentityExtractor, self).__init__()
        self.n_styles = n_styles
        self.latent_dim = latent_dim

        self.facenet = InceptionResnetV1(pretrained="vggface2").eval()
        self.activation = {}
        self.facenet.dropout.register_forward_hook(
            lambda _model, _input, output: self.activation.__setitem__(
                "dropout", output
            )
        )

        self.linear = nn.Linear(in_features=1792, out_features=n_styles * latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        with torch.no_grad():
            self.facenet(x)

        z = self.activation["dropout"].squeeze()
        z = self.linear(z)
        z = torch.reshape(z, shape=(batch_size, self.n_styles, -1))
        return z
