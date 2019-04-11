import numpy as np
import shutil, os

from imgaug import augmenters as iaa
import cv2

import csv
from shutil import copyfile
import random



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
                shutil.copy(path_to_file_name_src, path_to_file_name_dst)

    #shutil.rmtree(src_path)


# transformations
seq = iaa.Sequential(
    [
        # iaa.Fliplr(0.5), # horizontally flip 50% of the images

        iaa.Affine(
            scale={"x": (0.7, 1.0), "y": (0.7, 1.0)},
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15),
            shear=(-7, 7),
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=255,  # (0, 255), # if mode is constant, use a cval between 0 and 255
            mode='constant'
        ),

        iaa.SomeOf((0, 3),
            [
                iaa.GaussianBlur((0, 0.95)),  # blur images with a sigma of 0 to 3.0
                iaa.Sharpen(alpha=(0, 0.10), lightness=(0.85, 1.25)),  # sharpen images
                #iaa.Emboss(alpha=(0, 0.10), strength=(0, 0.3)), # emboss images
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025 * 255), per_channel=False),
                # add gaussian noise to images
                #iaa.CoarseDropout((0.05, 0.1), size_percent=(0.50, 0.70), per_channel=False),
                # iaa.Add((-25, 25), per_channel=False), # change brightness of images (by -10 to 10 of original value)
                #iaa.Multiply((0.75, 1.25), per_channel=False),
                # iaa.ContrastNormalization((0.85, 1.15), per_channel=False), # improve or worsen the contrast
                # distorsion
                #iaa.ElasticTransformation(alpha=(0.5, 2.0), sigma=0.25),
            ]
        )

    ],
    random_order=True
)


def preprocess_train(img):
    # print(img.shape, img.max(), img.min())

    img = img.astype(np.uint8)
    img = seq.augment_image(img)

    img = img.astype(np.float32)
    img /= 255.
    #img -= 0.5
    #img *= 2.

    return img

def DataGenerator(src_path, do_augment, batch_size=16,
            input_size=(40, 40)):
    all_file_list = []
    for path, subdirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.ppm'):
                all_file_list.append(os.path.join(path, file))

    random.shuffle(all_file_list)
    step_size = int(len(all_file_list) / batch_size)


    batch_labels = np.empty(shape=[0, 0])
    batch_imgs = np.zeros(shape=(batch_size,) + input_size + (3,))

    for idx in range(len(all_file_list)):
        file_name = all_file_list[idx]
        label_value = file_name.split('/')
        batch_labels = np.append(batch_labels, int(label_value[-1][0:5]))
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        if do_augment:
            img = seq.augment_image(img)

        batch_imgs[idx] = img
        if len(batch_imgs) % batch_size == 0:
            step_size -= 1
            batch_imgs = batch_imgs / 255.
            yield batch_imgs, batch_labels
        idx += 1

        if step_size == 0:
            break


def AugmentImages(path_to_ds, output_path, img_size, augment_num):
    batch_imgs, batch_labels = DataGenerator(path_to_ds, True)

    dir_names = os.listdir(path_to_ds)
    for dir_name in dir_names:
        path_to_data_set = '{}/{}'.format(path_to_ds, dir_name)
        img_names = os.listdir(path_to_data_set)

        output_path_dir = '{}/{}/'.format(output_path, dir_name)
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)

        csv_file_name = '{}/{}/{}'.format(output_path, dir_name, img_names[-1])
        in_csv = '{}/{}'.format(path_to_data_set, img_names[-1])
        copyfile(in_csv, csv_file_name)
        for img_name in img_names:
            #last file in img_names is csv
            if img_name != img_names[-1]:
                img_path = '{}/{}'.format(path_to_data_set, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                cv2.imwrite(output_path_dir + img_name, img)

                aug_idx = 0
                for i in range(augment_num):
                    aug_img = seq.augment_image(img)
                    file, ext = os.path.splitext(img_name)
                    final_img_name = '{}{}_aug_{}{}'.format(output_path_dir, file, aug_idx, ext)
                    aug_img_name = '{}_aug_{}{}'.format(file, aug_idx, ext)

                    line = [aug_img_name, img_size[0], img_size[1], 0, 0, 0, 0, int(dir_name)]
                    with open(csv_file_name, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=';')
                        writer.writerow(line)

                    cv2.imwrite(final_img_name, aug_img)
                    aug_idx += 1




#nameing convention is as follows: existing: ...00000_00028, 00000_00029; augmented: 00000_00030
#ALPHA VERSION
def AugmentImages_incr(path_to_ds):
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