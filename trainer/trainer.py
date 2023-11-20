import torch
import torch.backends.mps as mps
import torch.nn.functional as F

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "mps" if mps.is_available() else "cpu"
# )
device = "cpu"


class Trainer:
    def __init__(
        self,
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
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.generator = model_g.to(device)
        self.discriminator = model_d.to(device)
        self.loss_function = loss_function.to(device)

        # TODO: Clean this up somewhere else
        self.gan_loss_function = F.binary_cross_entropy

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_scheduler = lr_scheduler

        self.train_config = train_config
        self.validation_config = self.train_config["validation"]

        self.num_epochs = train_config["num_epochs"]
        self.start_epoch = 1
        self.log_step = train_config["log_step"]
        self.checkpoint_dir = train_config["checkpoint_dir"]
        self.save_period = train_config["save_period"]

        self.device = device
        self.logger = logger

    def train(self):
        self.generator.train()

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self._train_one_epoch(epoch)

            if (epoch + 1) % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_one_epoch(self, epoch: int):
        self.logger.info(f"Start training epoch {epoch}...")
        for batch_idx, item in enumerate(self.train_dataloader):
            masked_image = item["masked_image"].to(self.device)
            unmasked_image = item["unmasked_image"].to(self.device)
            identity_image = item["identity_image"].to(self.device)

            generated_unmasked_image = self.generator(masked_image, identity_image)

            loss, loss_dict = self.loss_function(
                unmasked_image,
                generated_unmasked_image,
                identity_image,
                self.discriminator,
                self.optimizer_d,
            )

            self.optimizer_g.zero_grad()
            loss.backward()
            self.optimizer_g.step()

            if batch_idx % self.log_step == 0:
                self.logger.info(
                    f"EPOCH {epoch} BATCH {batch_idx}: "
                    + f"total_loss={loss.item():.3f} "
                    + f"content_loss={loss_dict['content']:.3f} "
                    + f"perceptual_loss={loss_dict['perceptual']:.3f} "
                    + f"identity_loss={loss_dict['identity']:.3f} "
                    + f"generator_loss={loss_dict['adversarial']:.3f}"
                )

        self.lr_scheduler.step()

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
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

        filename = f"{self.checkpoint_dir}/{self.generator.__class__.__name__}-epoch{epoch}.pth"
        torch.save(state, filename)
        self.logger.info(f"Saved checkpoint to: {filename}")

        if is_best:
            best_path = str(self.checkpoint_dir / "checkpoint_best.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saved current best to: {best_path}")

    def _load_checkpoint(self, resume_config):
        """Load checkpoint"""
        resume_path = resume_config
        self.logger.info(f"Loading checkpoint from {resume_path}...")

        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1

        if checkpoint["config"]["architecture"] != self.train_config["architecture"]:
            self.logger.warning(
                "Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
