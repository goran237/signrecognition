from src.utils.data.process.DataSetPreparation import AugmentImages
import src.utils.data.process.DataSetPreparation as dsp
from src.utils.data.importer.DataImporter import import_training_data,import_test_data
from src.utils.data.process.DataProcessor import preprocess_training_images,preprocess_test_images
from src.train.Trainer import perform_train
import sys
from ast import literal_eval

from src.test.Tester import perform_test
from src.train.train import TrainModel
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as pat

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print('aaaaaaaaaaaaaaa')

from keras import backend as K


def main():
    #import_training_data()
    #import_test_data()
    #preprocess_training_images()
    #preprocess_test_images()
    #perform_train()
    #perform_test()
    #AugmentImages(sys.argv[1], sys.argv[2], literal_eval(sys.argv[3]), int(sys.argv[4]))
    #DataSetPreparation('e:/deephunter/data_set/Images/', 0.1)

    datagen_train = ImageDataGenerator(preprocessing_function=dsp.preprocess_train)
    datagen_valid = ImageDataGenerator(preprocessing_function=dsp.preprocess_train)

    nb_batch = 32
    input_size = (64, 64)
    generator_train = datagen_train.flow_from_directory('e:/deephunter/data_set/Images/data/train/', target_size=input_size, color_mode='grayscale')

    steps_train = generator_train.n // nb_batch
    generator_valid = datagen_valid.flow_from_directory('e:/deephunter/data_set/Images/data/valid/', target_size=input_size, color_mode='grayscale')
    steps_valid = generator_valid.n // nb_batch


    plt.rcParams['figure.figsize'] = (8, 8)
    imgs, labels = generator_train.next()
    imgs.shape, imgs.max(), imgs.min()
    for i, img in enumerate(imgs):
        plt.subplot(6, 6, 1 + i)
        plt.imshow(imgs[i, ..., 0], cmap=plt.get_cmap('gray'))
        plt.axis('off')

    '''plt.rcParams['figure.figsize'] = (8, 8)
    imgs, labels = generator_valid.next()
    imgs.shape, imgs.max(), imgs.min()
    for i, img in enumerate(imgs):
        plt.subplot(6, 6, 1 + i)
        plt.imshow(imgs[i, ..., 0], cmap=plt.get_cmap('gray'))
        plt.axis('off')'''



    TrainModel(generator_train, steps_train, generator_valid, steps_valid)


if __name__ == '__main__':
    main()
