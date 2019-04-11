from keras.layers import Input, Conv2D, Reshape, AvgPool2D, MaxPool2D, \
Concatenate, BatchNormalization, Deconv2D, Activation, Add, Dense, Dropout

from keras.models import Model, load_model

from keras.optimizers import Adam, SGD

from keras.objectives import binary_crossentropy, mse, categorical_crossentropy

from keras.callbacks import ModelCheckpoint

from src.train.keras_spatial_bias import ConcatSpatialCoordinate

from keras import backend as K

import tensorflow as tf

from src.train.clr_callback import CyclicLR

from src.utils.data.process.DataSetPreparation import preprocess_train


def ModelArchitecture(input_size, class_names):

    t_inp = Input(shape=input_size + (1,), name='input')

    t = ConcatSpatialCoordinate()(t_inp)

    t = Conv2D(16, (3, 3), padding='same', activation='relu')(t)
    t = MaxPool2D()(t)

    t = Conv2D(32, (3, 3), padding='same', activation='relu')(t)
    t = Conv2D(64, (1, 1), padding='same', activation='relu')(t)
    t = MaxPool2D()(t)

    t = Conv2D(64, (3, 3), padding='same', activation='relu')(t)
    t = Conv2D(128, (1, 1), padding='same', activation='relu')(t)
    t = MaxPool2D()(t)

    t = Conv2D(128, (3, 3), padding='same', activation='relu')(t)
    t = Conv2D(256, (1, 1), padding='same', activation='relu')(t)

    t = Conv2D(128, (3, 3), padding='same', activation='relu')(t)
    t = Conv2D(256, (1, 1), padding='same', activation='relu')(t)
    # t = MaxPool2D()(t)

    t = AvgPool2D((7, 5))(t)

    t = Reshape((256,))(t)
    t = Dropout(0.35)(t)

    t_out = Dense(len(class_names), activation='softmax', name='output')(t)
    model = Model(t_inp, t_out)

    model.summary()

    t_cc = model.get_layer('concat_spatial_coordinate_1').output
    f = K.function([t_inp], [t_cc])
    #out = f([imgs[0:1]])[0]
    #for i in range(3):
    #    plt.subplot(1, 3, 1 + i)
    #    plt.imshow(out[0, :, :, i])

    model.compile(loss=categorical_crossentropy, optimizer=Adam(1e-3), metrics=['accuracy'])
    return model

def TrainModel(generator_train, steps_train, generator_valid, steps_valid):

    clr_triangular = CyclicLR(mode='triangular', base_lr=4e-5, max_lr=3e-4, step_size=200)
    save = ModelCheckpoint('e:/deephunter/saved/weights.properchar4-1.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.h5', period=10, verbose=1)

    nb_epochs = int(1e9)

    input_size = (64, 64)
    #class_names = len(generator_train.class_indices)
    class_indices = generator_valid.class_indices
    class_names = {}

    for key, val in class_indices.items():
        class_names[val] = key

    model = ModelArchitecture(input_size, class_names)

    print("##########################################")

    model.fit_generator(generator_train, steps_train, epochs=nb_epochs, validation_data=generator_valid, validation_steps=steps_valid, callbacks=[save, clr_triangular])