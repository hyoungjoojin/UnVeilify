import os
import random
from typing import Dict, Tuple

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MaskPairImageDataset(Dataset):
    def __init__(
        self,
        masked_image_dir: str,
        identity_image_dir: str,
        unmasked_image_dir: str,
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

        self.masked_image_list = os.listdir(masked_image_dir)
        self.identity_image_list = {
            identity: os.listdir(os.path.join(identity_image_dir, identity))
            for identity in os.listdir(identity_image_dir)
        }

        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, idx: int) -> Dict:
        label = self.masked_image_list[idx]
        identity = label.split(".")[0]

        masked_image_path = os.path.join(self.masked_image_dir, label)
        unmasked_image_path = os.path.join(self.unmasked_image_dir, label)
        identity_image_path = os.path.join(
            self.identity_image_dir,
            identity,
            random.choice(self.identity_image_list[identity]),
        )

        masked_image = cv2.imread(masked_image_path)
        unmasked_image = cv2.imread(unmasked_image_path)
        identity_image = cv2.imread(identity_image_path)

        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        unmasked_image = cv2.cvtColor(unmasked_image, cv2.COLOR_BGR2RGB)
        identity_image = cv2.cvtColor(identity_image, cv2.COLOR_BGR2RGB)

        masked_image = torch.permute(torch.from_numpy(masked_image), (2, 0, 1)).float()
        unmasked_image = torch.permute(
            torch.from_numpy(unmasked_image), (2, 0, 1)
        ).float()
        identity_image = torch.permute(
            torch.from_numpy(identity_image), (2, 0, 1)
        ).float()

        masked_image = self.transform(masked_image)
        unmasked_image = self.transform(unmasked_image)
        identity_image = self.transform(identity_image)

        return {
            "masked_image": identity_image,
            "identity_image": unmasked_image,
            "unmasked_image": identity_image,
        }

    def __len__(self) -> int:
        return len(self.masked_image_list)

    def paired_transform(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform for paired images"""
        # TODO
        return img1, img2
