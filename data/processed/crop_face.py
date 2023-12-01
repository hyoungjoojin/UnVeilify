import os
from typing import Tuple

import cv2
import face_recognition
import numpy as np
from tqdm import tqdm

MASKED_IMAGE_DIR = "../raw/masked_image"
IDENTITY_IMAGE_DIR = "../raw/identity_image"
UNMASKED_IMAGE_DIR = "../raw/unmasked_image"


def main():
    # Process masked and unmasked images
    for label in tqdm(os.listdir(UNMASKED_IMAGE_DIR)):
        masked_image = cv2.imread(os.path.join(MASKED_IMAGE_DIR, label))
        unmasked_image = cv2.imread(os.path.join(UNMASKED_IMAGE_DIR, label))
        if masked_image.shape != unmasked_image.shape:
            print("Shapes do not match!")
            exit()

        top, right, bottom, left = get_face_coordinates(unmasked_image)
        masked_image = masked_image[top:bottom, left:right]
        unmasked_image = unmasked_image[top:bottom, left:right]

        masked_image = cv2.resize(masked_image, (512, 512))
        unmasked_image = cv2.resize(unmasked_image, (512, 512))

        cv2.imwrite(f"./masked_image/{label}", masked_image)
        cv2.imwrite(f"./unmasked_image/{label}", unmasked_image)

    # Process identity images
    for label in tqdm(os.listdir(IDENTITY_IMAGE_DIR)):
        os.mkdir(f"./identity_image/{label}")

        for sublabel in os.listdir(os.path.join(IDENTITY_IMAGE_DIR, label)):
            identity_image = cv2.imread(
                os.path.join(IDENTITY_IMAGE_DIR, label, sublabel)
            )

            identity_image = cv2.resize(identity_image, (512, 512))

            cv2.imwrite(f"./identity_image/{label}/{sublabel}", identity_image)


def get_face_coordinates(
    image: np.ndarray,
    padding: int = 50,
    scaling_factor: int = 4,
) -> Tuple[int, int, int, int]:
    """Detect faces from given input video frames

    Given a list of video frames, detect the location of faces in each frame. The
    open source library face_recognition (https://github.com/ageitgey/face_recognition)
    is used.

    Args:
        image (np.ndarray): The input image.
        padding (int): Padding around the detected face.
        scaling_factor (int): Scaling factor of processing frames, larger scaling factor
            increases processing time while decreasing accuracy.

    Returns:
        A list of tuples indicating the coordinates of the face detected. Each tuple goes
        by the form of (top, right, bottom, left).

    """
    image = cv2.resize(image, (0, 0), fx=1 / scaling_factor, fy=1 / scaling_factor)
    face_location = face_recognition.face_locations(image, model="cnn")[0]

    top, right, bottom, left = face_location
    top = scaling_factor * top - padding
    right = scaling_factor * right + padding  # type: ignore
    bottom = scaling_factor * bottom + padding  # type: ignore
    left = scaling_factor * left - padding

    return (top, right, bottom, left)


if __name__ == "__main__":
    main()
