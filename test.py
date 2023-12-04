import argparse
import os

import torch
import torch.backends.mps as mps
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dataloader.dataloader import MaskPairImageDataLoader
from model.mask_remover import MaskRemover

device = torch.device(
    "cuda" if torch.cuda.is_available else "mps" if mps.is_available() else "cpu"
)

denormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


def main(args):
    dataset_path = args.dataset
    checkpoint_path = args.checkpoint
    output_resolution = args.resolution
    output_path = args.output

    dataloader = MaskPairImageDataLoader(
        masked_image_dir=os.path.join(dataset_path, "masked_image"),
        unmasked_image_dir=os.path.join(dataset_path, "unmasked_image"),
        identity_image_dir=os.path.join(dataset_path, "identity_image"),
        batch_size=16,
        image_resolution=output_resolution,
    )

    model = MaskRemover(output_resolution=output_resolution)
    model.load_state_dict(
        torch.load(checkpoint_path)["generator_state_dict"],
        strict=False,
    )

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for item in tqdm(dataloader):
            masked_image = item["masked_image"].to(device)
            identity_image = item["identity_image"].to(device)

            label = item["identity"]

            output = model(masked_image, identity_image)
            output = denormalize(output)

            for i in range(masked_image.shape[0]):
                image_to_save = to_pil_image(
                    output[i],
                    mode="RGB",
                )
                image_to_save.save(os.path.join(output_path, label[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("-r", "--resolution", default=256, help="Output resolution")
    parser.add_argument(
        "-d",
        "--dataset",
        default="./data/processed/test",
        help="Path to test dataset. Expected to have (masked_image/unmasked_image/identity_image) as subpaths.",
    )
    parser.add_argument(
        "-o", "--output", default="./output", help="Path for output images."
    )

    args = parser.parse_args()
    main(args)
