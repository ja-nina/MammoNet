# import numpy as np
# from imgaug import augmenters as iaa
# from PIL import Image

# class ImgAugTransform:
#     '''
#     Transformations taken from: https://www.kaggle.com/code/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb
#     '''
#     def __init__(self):
#         self.aug = iaa.Sequential(
#             [
#                 iaa.Fliplr(0.5),
#                 iaa.Flipud(0.2),
#                 iaa.Sometimes(0.5, iaa.Affine(
#                     scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#                     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#                     rotate=(-10, 10),
#                     shear=(-5, 5),
#                     order=1,  # Changed to a single integer
#                     cval=(0, 255),  # Changed to a tuple of two integers
#                     mode='constant'  # Changed to a string
#                 )),
#                 iaa.SomeOf((0, 5),
#                     [
#                         iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
#                         iaa.OneOf([
#                             iaa.GaussianBlur((0, 1.0)),
#                             iaa.AverageBlur(k=(3, 5)),
#                             iaa.MedianBlur(k=(3, 5)),
#                         ]),
#                         iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
#                         iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
#                         iaa.SimplexNoiseAlpha(iaa.OneOf([
#                             iaa.EdgeDetect(alpha=(0.5, 1.0)),
#                             iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
#                         ])),
#                         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
#                         iaa.OneOf([
#                             iaa.Dropout((0.01, 0.05), per_channel=0.5),
#                             iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
#                         ]),
#                         iaa.Invert(0.01, per_channel=True),
#                         iaa.Add((-2, 2), per_channel=0.5),
#                         iaa.AddToHueAndSaturation((-1, 1)),
#                         iaa.OneOf([
#                             iaa.Multiply((0.9, 1.1), per_channel=0.5),
#                             iaa.FrequencyNoiseAlpha(
#                                 exponent=(-1, 0),
#                                 first=iaa.Multiply((0.9, 1.1), per_channel=True),
#                                 second=iaa.ContrastNormalization((0.9, 1.1))
#                             )
#                         ]),
#                         iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
#                         iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
#                         iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1)))
#                     ],
#                     random_order=True
#                 )
#             ],
#             random_order=True
#         )

#     def __call__(self, img):
#         img = np.array(img)
#         img = self.aug(image=img)
#         return Image.fromarray(img)

import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
import os 


class ImgAugTransform:
    '''
    Transformations taken from: https://www.kaggle.com/code/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb
    '''
    def __init__(self):
        self.aug = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2),
                iaa.Sometimes(0.5, iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),
                    shear=(-5, 5),
                    order=1, 
                    cval=(0, 255),
                    mode='constant'  
                ))
            ],
            random_order=True
        )

    def __call__(self, img):
        img = np.array(img)
        img = self.aug(image=img)
        augmented_image = np.clip(img, 0, 255).astype(np.uint8)
        return img
    

            
        
