import numpy as np
import albumentations as A
import cv2
from PIL import Image


class ImageAugmentations:
    """
    Transformations inspired by: https://www.kaggle.com/code/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb
    """

    def __init__(self):
        self.aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.OneOf(
                    [
                        A.Affine(
                            scale=(0.9, 1.1),
                            mode=1,
                            cval=(0, 255),
                            # decided against rotation, because of resulting mask
                        ),
                        A.SafeRotate(
                            p=1,
                            limit=(-20, 20),
                            rotate_method="ellipse",
                            interpolation=cv2.INTER_AREA,
                        ),
                    ],
                    p=0.7,
                ),
                A.SomeOf(
                    [
                        A.Superpixels(p_replace=(0, 0.6), n_segments=(20, 200), p=0.5),
                        A.OneOf(
                            [
                                A.GaussianBlur(blur_limit=(1, 3.0), p=1.0),
                                A.MedianBlur(blur_limit=3, p=1.0),
                                A.MotionBlur(blur_limit=3, p=1.0),
                            ],
                            p=1.0,
                        ),
                        A.Sharpen(p=1.0),
                        A.Emboss(p=1.0),
                        A.InvertImg(p=0.01),
                        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
                        A.HueSaturationValue(
                            hue_shift_limit=(-1, 1), sat_shift_limit=(-1, 1), val_shift_limit=(-1, 1), p=1.0
                        ),
                        A.OneOf(
                            [
                                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0),
                                A.RandomBrightnessContrast(
                                    brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0
                                ),
                            ],
                            p=1.0,
                        ),
                        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                        A.Perspective(scale=(0.01, 0.1), p=0.5, keep_size=False),
                    ],
                    n=5,
                    p=1.0,
                ),
            ],
            p=1.0,
        )

    def __call__(self, img):
        img = np.array(img)
        augmented = self.aug(image=img)
        return Image.fromarray(augmented["image"])
