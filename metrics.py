import argparse
import os

import pandas as pd
import piq
from lpips import LPIPS
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def main(args):
    labels = os.listdir(args.d1)
    df = pd.DataFrame(labels, columns=["label"])
    df["psnr"] = 0
    df["ssim"] = 0
    df["lpips"] = 0

    lpips = LPIPS()
    for i, label in tqdm(enumerate(labels), total=len(labels)):
        img1 = pil_to_tensor(
            Image.open(os.path.join(args.d1, label)).convert("RGB")
        ).unsqueeze(dim=0)
        img2 = pil_to_tensor(
            Image.open(os.path.join(args.d2, label)).convert("RGB")
        ).unsqueeze(dim=0)

        df.loc[i, "psnr"] = piq.psnr(img1, img2, data_range=255).item()
        df.loc[i, "ssim"] = piq.ssim(img1, img2, data_range=255).item()
        df.loc[i, "lpips"] = lpips(img1, img2).item()

    df.to_csv("./results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d1", required=True, help="Directory one.")
    parser.add_argument("-d2", required=True, help="Directory two.")
    args = parser.parse_args()
    main(args)
