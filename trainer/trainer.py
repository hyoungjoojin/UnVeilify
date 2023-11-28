import os
from typing import Union

import torch
import torch.backends.mps as mps
import torchvision.transforms as transforms
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import to_pil_image

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if mps.is_available() else "cpu"
)


class Trainer:
    def __init__(
        self,
        project_name,
        train_dataloader,
        validation_dataloader,
        model_g,
        model_d,
        loss_function,
        optimizer_g,
        optimizer_d,
        lr_scheduler,
        train_config,
        logger,
    ) -> None:
        self.project_name = project_name
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.generator = model_g.to(device)
        self.discriminator = model_d.to(device)
        self.loss_function = loss_function.to(device)

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_scheduler = lr_scheduler

        self.train_config = train_config
        self.validation_config = self.train_config["validation"]

        self.num_epochs = train_config["num_epochs"]
        self.start_epoch = 1
        self.log_step = train_config["log_step"]
        self.checkpoint_dir = os.path.join(train_config["checkpoint_dir"], project_name)
        self.save_period = train_config["save_period"]

        self.visualization = train_config["visualization"]

        self.device = device
        self.logger = logger

        self.denormalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        self.setup_directory()
        if train_config["use_tensorboard"]:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.checkpoint_dir, "logs")
            )
        else:
            self.writer = None

    def setup_directory(self):
        if os.path.exists(self.checkpoint_dir):
            self.logger.warning(f"Path {self.checkpoint_dir} already exists.")

            checkpoint = get_most_recent_file(
                os.path.join(self.checkpoint_dir, "checkpoints")
            )
            if checkpoint is None:
                return

            start_epoch = self.load_checkpoint(checkpoint)
            self.start_epoch = start_epoch

            self.logger.info(
                f"Found checkpoint {checkpoint}. "
                + f"Starting training from epoch {self.start_epoch}."
            )
            return

        os.mkdir(self.checkpoint_dir)
        os.mkdir(os.path.join(self.checkpoint_dir, "logs"))
        os.mkdir(os.path.join(self.checkpoint_dir, "visualization"))
        os.mkdir(os.path.join(self.checkpoint_dir, "checkpoints"))

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.generator.train()
            self.discriminator.train()

            self.train_one_epoch(epoch)

            if (epoch + 1) % self.save_period == 0:
                self.generator.eval()
                self.discriminator.eval()

                self.save_checkpoint(epoch)

                if self.visualization:
                    self.visualize(filename=f"{self.project_name}-epoch{epoch}.jpg")

    def train_one_epoch(self, epoch: int):
        self.logger.info(f"Epoch {epoch}")
        for batch_idx, item in enumerate(self.train_dataloader):
            masked_image = item["masked_image"].to(self.device)
            unmasked_image = item["unmasked_image"].to(self.device)
            identity_image = item["identity_image"].to(self.device)

            identity_features, generated_unmasked_image = self.generator(
                masked_image, identity_image
            )

            loss, loss_dict = self.loss_function(
                unmasked_image,
                generated_unmasked_image,
                identity_image,
                identity_features,
                self.discriminator,
                self.optimizer_d,
            )

            self.optimizer_g.zero_grad()
            loss.backward()
            self.optimizer_g.step()

            if batch_idx % self.log_step == 0:
                self.logger.info(
                    f"[epoch: {epoch} batch: {batch_idx}] "
                    + f"total_loss={loss.item():.3f} "
                    + f"content_loss={loss_dict['content']:.3f} "
                    + f"perceptual_loss={loss_dict['perceptual']:.3f} "
                    + f"identity_loss={loss_dict['identity']:.3f} "
                    + f"generator_loss={loss_dict['adversarial']:.3f}"
                )

                if self.writer is not None:
                    self.writer.add_scalar("loss/train", loss)

        self.lr_scheduler.step()

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save checkpoints

        Save checkpoints for a given model with the option of saving the checkpoint
        as the best performing model.

        Args:
            epoch (int): Current epoch
            is_best (bool): If true, additionally save the model to model_best.pth

        """
        state = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optim_g_state_dict": self.optimizer_g.state_dict(),
            "optim_d_state_dict": self.optimizer_d.state_dict(),
            "config": self.train_config,
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, "checkpoints", f"{self.project_name}-epoch{epoch}.pth"
        )
        torch.save(state, checkpoint_path)
        self.logger.info(f"Saved checkpoint to: {checkpoint_path}")

        if is_best:
            best_path = str(self.checkpoint_dir / "checkpoint_best.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saved current best to: {best_path}")

    def load_checkpoint(self, checkpoint_path) -> int:
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optim_g_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optim_d_state_dict"])
        return checkpoint["epoch"] + 1

    def visualize(self, filename) -> None:
        visualization_path = os.path.join(self.checkpoint_dir, "visualization")
        with torch.no_grad():
            item = next(iter(self.validation_dataloader))
            masked_image = item["masked_image"][0].to(self.device).unsqueeze(dim=0)
            unmasked_image = item["unmasked_image"][0].to(self.device).unsqueeze(dim=0)
            identity_image = item["identity_image"][0].to(self.device).unsqueeze(dim=0)

            _, generated_output = self.generator(masked_image, identity_image)

            masked_image = masked_image.squeeze().detach().cpu()
            unmasked_image = unmasked_image.squeeze().detach().cpu()
            generated_output = generated_output.squeeze().detach().cpu()

            masked_image = self.denormalize(masked_image)
            unmasked_image = self.denormalize(unmasked_image)
            generated_output = self.denormalize(generated_output)

            image_to_save = torch.cat(
                [masked_image, unmasked_image, generated_output], dim=2
            )
            image_to_save = to_pil_image(
                image_to_save,
                mode="RGB",
            )
            image_to_save.save(os.path.join(visualization_path, filename))


def get_most_recent_file(path: str) -> Union[None, str]:
    if os.path.exists(path) == False:
        return None

    files = os.listdir(path)

    paths = [os.path.join(path, basename) for basename in files]
    if len(paths) == 0:
        return None

    return max(paths, key=os.path.getctime)
