import numpy as np
import shutil, os

import imgaug as ia
from imgaug import augmenters as iaa
import cv2

from PIL import Image, ImageDraw, ImageFont, ImageFilter

def DataSetPreparation(src_path, valid_ratio):
    dir_names = os.listdir(src_path)
    head, tail = os.path.split(src_path)
    dst = '{}/{}'.format(head, 'data')
    for dir_name in dir_names:

        print('Processing:', dir_name)
        path_to_dir_name = '{}/{}'.format(src_path, dir_name)
        file_names = os.listdir(path_to_dir_name)
        nb = len(file_names)
        nb_valid = int(valid_ratio * nb)

        np.random.shuffle(file_names)
        file_names_valid = file_names[:nb_valid]
        file_names_train = file_names[nb_valid:]

        for phase, file_names_phase in zip(['train', 'valid'], [file_names_train, file_names_valid]):
            for file_name in file_names_phase:
                path_to_file_name_src = '{}/{}'.format(path_to_dir_name, file_name)
                path_to_dir_name_dst = '{}/{}/{}'.format(dst, phase, dir_name)
                try:
                    os.makedirs(path_to_dir_name_dst)
                except:
                    pass

                path_to_file_name_dst = '{}/{}'.format(path_to_dir_name_dst, file_name)
                shutil.move(path_to_file_name_src, path_to_file_name_dst)

    shutil.rmtree(src_path)


# transformations
seq = iaa.Sequential(
    [
        # iaa.Fliplr(0.5), # horizontally flip 50% of the images

        iaa.Affine(
            scale={"x": (0.5, 1.0), "y": (0.5, 1.0)},
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15),
            shear=(-7, 7),
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=255,  # (0, 255), # if mode is constant, use a cval between 0 and 255
            mode='constant'
        ),

        iaa.SomeOf((0, 6),
            [
                iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma of 0 to 3.0
                iaa.Sharpen(alpha=(0, 0.10), lightness=(0.85, 1.25)),  # sharpen images
                # iaa.Emboss(alpha=(0, 0.10), strength=(0, 0.3)), # emboss images
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025 * 255), per_channel=False),
                # add gaussian noise to images
                iaa.CoarseDropout((0.05, 0.1), size_percent=(0.50, 0.70), per_channel=False),
                # iaa.Add((-25, 25), per_channel=False), # change brightness of images (by -10 to 10 of original value)
                iaa.Multiply((0.75, 1.25), per_channel=False),
                # iaa.ContrastNormalization((0.85, 1.15), per_channel=False), # improve or worsen the contrast
                # distorsion
                iaa.ElasticTransformation(alpha=(0.5, 2.0), sigma=0.25),
            ]
        )

    ],
    random_order=True
)

def AugmentImages(path_to_ds):
    dir_names = os.listdir(path_to_ds)

    for dir_name in dir_names:
        path_to_data_set = '{}/{}'.format(path_to_ds, dir_name)
        img_names = os.listdir(path_to_data_set)
        idx = 0
        first_part_old = ""
        for img_name in img_names:
            first_part_img_name = img_name.split('_')
            first_part = first_part_img_name[0] + '_'
            if (first_part_old != first_part):
                res = list(filter(lambda x: first_part in x, img_names))
                img_idx = res[-1]
                img_idx = img_idx.split('_')
                img_idx = img_idx[1].split('.')
                img_idx = int(img_idx[0]) + 1
                idx = img_idx
                first_part_old = first_part


            img_path = '{}/{}'.format(path_to_data_set, img_name)
            img = cv2.imread(img_path)
            img = seq.augment_image(img)
            file, ext = os.path.splitext(img_path)

            cv2.imwrite('e:/deephunter/signrecognition/src/utils/data/GTSRB/Final_Training/bla/' + first_part + "%05d" % idx + '.ppm', img)
            print(first_part + "%05d" % idx + '.ppm')
            idx += 1

        print(path_to_data_set)
        break