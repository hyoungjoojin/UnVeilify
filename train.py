from importlib import import_module

from torchvision.transforms.functional import torch

import utils
from trainer.trainer import Trainer

packages = {
    "dataloader_module": import_module(".dataloader", "dataloader"),
    "model_module": import_module("model"),
    "loss_module": import_module("model", "losses"),
    "optimizer_module": import_module("torch.optim"),
    "lr_scheduler_module": import_module("torch.optim.lr_scheduler"),
}


def main(config):
    project_name = config["name"]

    if config["deterministic"]:
        utils.set_seed(config["seed"])

    logger = config.get_logger("train")

    train_dataloader = config.build("train_dataloader", packages["dataloader_module"])

    validation_dataloader = config.build(
        "validation_dataloader", packages["dataloader_module"]
    )

    logger.info(f"Dataloader build success.")

    generator = config.build("model", packages["model_module"], submodule="generator")
    discriminator = config.build(
        "model", packages["model_module"], submodule="discriminator"
    )
    logger.info(f"Model build success.\n{generator}\n{discriminator}")

    loss_function = config.build("loss", packages["loss_module"])
    logger.info("Loss build success.")

    optimizer_g = config.build(
        "optimizer",
        packages["optimizer_module"],
        generator.parameters(),
        submodule="optim_g",
    )
    optimizer_d = config.build(
        "optimizer",
        packages["optimizer_module"],
        discriminator.parameters(),
        submodule="optim_d",
    )

    logger.info("Optimizer build success.")

    lr_scheduler = config.build(
        "lr_scheduler", packages["lr_scheduler_module"], optimizer_g
    )

    if config["lr_scheduler"]["use_warmup"]:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer_g,
            lr_lambda=lambda current_step: 1
            / (10 ** (float(config["lr_scheduler"]["warmup_epochs"] - current_step))),
        )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer_g,
            [warmup_scheduler, lr_scheduler],
            [config["lr_scheduler"]["warmup_epochs"]],
        )

    if lr_scheduler is not None:
        logger.info("LR scheduler build success.")

    trainer = Trainer(
        project_name,
        train_dataloader,
        validation_dataloader,
        generator,
        discriminator,
        loss_function,
        optimizer_g,
        optimizer_d,
        lr_scheduler,
        config["trainer"],
        logger,
    )
    logger.info("Trainer build success.")

    logger.info("Start training. Good luck !!")
    trainer.train()


if __name__ == "__main__":
    config_file, args = utils.parse_argument()
    config = utils.ConfigParser(config_file, args)
    main(config)
