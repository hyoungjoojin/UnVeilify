from torch.utils.data import DataLoader

__all__ = ["MaskPairImageDataLoader"]

from .dataset import MaskPairImageDataset


class MaskPairImageDataLoader(DataLoader):
    def __init__(
        self,
        masked_image_dir: str,
        identity_image_dir: str,
        unmasked_image_dir: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 2,
    ) -> None:
        self.dataset = MaskPairImageDataset(
            masked_image_dir, identity_image_dir, unmasked_image_dir
        )
        self.dataset_length = len(self.dataset)
        self.batch_size = batch_size

        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
