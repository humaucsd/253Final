import os

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from DataGen2 import ImageDataGeneratorV2
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from keras import backend as K
K.set_image_dim_ordering('th')

def AlexNet(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
        inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
        Convolution2D(128, 5, 5, activation="relu", name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096,6,6,activation="relu",name="dense_1")(dense_1)
        dense_2 = Convolution2D(4096,1,1,activation="relu",name="dense_2")(dense_1)
        dense_3 = Convolution2D(1000, 1,1,name="dense_3")(dense_2)
        prediction = Softmax4D(axis=1,name="softmax")(dense_3)
    else:
        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000,name='dense_3')(dense_3)
        prediction = Activation("softmax",name="softmax")(dense_3)


    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)

    return model



def get_gender_model(weight_dir):
    weights_path = weight_dir
    alex_model = AlexNet(weights_path, heatmap=False)


    #Pick the last desired layer of AlexNet:

    alex_out = alex_model.layers[-6].output

    #Make our own layers
    fc_gender1 = Dense(512, activation='relu', name='fc_gender1')(alex_out)
    fc_gender2 = Dense(2, activation='softmax', init='he_normal', name='fc_gender2')(fc_gender1)

    #Defining the model
    gender_model = Model(input = alex_model.input, output=fc_gender2)
    return gender_model


if __name__ == '__main__':
    train_data_dir = "Train"
    val_data_dir = "Validation"
    test_data_dir = "Test"
    nb_epoch = 2
    nb_sample_per_epoch = 32
    nb_validation_samples = 1000
    nb_test_samples = 1000
    gender_model = get_gender_model("alexnet_weights.h5")
    train_data = ImageDataGeneratorV2()
    train_data_flow = train_data.flow_from_directory(os.path.join(train_data_dir, "flickr"),
                                                     os.path.join(train_data_dir, "Pos.json"),
                                                     os.path.join(train_data_dir, "Neg.json"), output_type='gender')
    val_data = ImageDataGeneratorV2()
    val_data_flow = val_data.flow_from_directory(os.path.join(val_data_dir, "flickr"),
                                                 os.path.join(val_data_dir, "Pos.json"),
                                                 os.path.join(val_data_dir, "Neg.json"), output_type='gender')

    test_data = ImageDataGeneratorV2()
    test_data_flow = test_data.flow_from_directory(os.path.join(test_data_dir, "flickr"),
                                                   os.path.join(test_data_dir, "Pos.json"),
                                                   os.path.join(test_data_dir, "Neg.json"), output_type='gender')

    gender_model.fit_generator(
        train_data_flow,
        samples_per_epoch=nb_sample_per_epoch,
        nb_epoch=nb_epoch,
        validation_data=val_data_flow,
        nb_val_samples=nb_validation_samples
    )

    scores = gender_model.evaluate_generator(test_data_flow, nb_test_samples)

