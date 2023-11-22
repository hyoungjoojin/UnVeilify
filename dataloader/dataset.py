import os
import random
from typing import Dict, Tuple

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MaskPairImageDataset(Dataset):
    def __init__(
        self,
        masked_image_dir: str,
        identity_image_dir: str,
        unmasked_image_dir: str,
        image_resolution: int = 512,
    ) -> None:
        """Dataset for image pairs with masked and unmasked face.

        Expects the data folder to be in the form of:

        |- data
        |-- masked_image
        |--- A_1.png
        |--- A_1.png
        |--- B_2.png
        |--- ...
        |-- identity_image
        |--- A
        |---- 1.png
        |---- 2.png
        |---- 3.png
        |--- B
        |---- 1.png
        |---- 2.png
        |--- ...
        |-- unmasked_image
        |--- A_1.png
        |--- A_1.png
        |--- B_2.png
        |--- ...

        Args:
            masked_image_dir (str): Path to the masked images.
            identity_image_dir (str): Path to the identity images.
            unmasked_image_dir (str): Path to the unmasked images, should be paired
                with the masked_image_dir.
            transforms (transforms.Compose): Transforms object for data augmentation.


        """
        super(MaskPairImageDataset, self).__init__()

        self.masked_image_dir = masked_image_dir
        self.identity_image_dir = identity_image_dir
        self.unmasked_image_dir = unmasked_image_dir
        self.image_resolution = image_resolution

        self.masked_image_list = os.listdir(masked_image_dir)
        self.identity_image_list = {
            identity: os.listdir(os.path.join(identity_image_dir, identity))
            for identity in os.listdir(identity_image_dir)
        }

        self.augmentation = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=90),
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                transforms.Resize((image_resolution, image_resolution)),
            ]
        )

    def __getitem__(self, idx: int) -> Dict:
        label = self.masked_image_list[idx]
        if label[0] == "F":
            identity = label.split(".")[0]
        else:
            identity = label.split("_")[0]

        masked_image_path = os.path.join(self.masked_image_dir, label)
        unmasked_image_path = os.path.join(self.unmasked_image_dir, label)
        identity_image_path = os.path.join(
            self.identity_image_dir,
            identity,
            random.choice(self.identity_image_list[identity]),
        )

        masked_image = Image.open(masked_image_path).convert("RGB")
        unmasked_image = Image.open(unmasked_image_path).convert("RGB")
        identity_image = Image.open(identity_image_path).convert("RGB")

        masked_image = self.transform(masked_image)
        unmasked_image = self.transform(unmasked_image)
        identity_image = self.transform(identity_image)

        return {
            "masked_image": masked_image,
            "identity_image": identity_image,
            "unmasked_image": unmasked_image,
        }

    def __len__(self) -> int:
        return len(self.masked_image_list)

    def paired_augmentation(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform for paired images"""
        if random.random() < 0.5:
            img1 = np.flip(img1, 0)
            img2 = np.flip(img2, 0)

        if random.random() < 0.5:
            img1 = np.flip(img1, 1)
            img2 = np.flip(img2, 1)

        rot_k = random.choice([0, 1, 2, 3])
        img1 = np.rot90(img1, k=rot_k, axes=(0, 1))
        img2 = np.rot90(img2, k=rot_k, axes=(0, 1))
        return img1, img2
