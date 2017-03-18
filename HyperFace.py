
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import layers

from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from keras import backend as K
K.set_image_dim_ordering('th')

def AlexNet(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
        inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2), name = "max_1")(conv_1)
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


def get_hyperface_model(weight_dir):
    weights_path = weight_dir
    alex_model = AlexNet(weights_path, heatmap=False)

    #Pick the last desired layer of AlexNet:

    pool5 = alex_model.layers[-8].output


    #Make our own layers
    alex_out1 = Convolution2D(256, 4, 4, subsample=(4,4), activation="relu", name="conv1a")(alex_model.get_layer('max_1').output)

    alex_out2 = Convolution2D(256, 2, 2, subsample=(2,2), activation="relu", name="conv3a")(alex_model.get_layer('conv_3').output)


    #concatenate outputs
    merged_output = layers.merge([alex_out1, alex_out2, pool5],mode='concat',concat_axis=0)
    conv_all = Convolution2D(192, 1, 1, activation='relu', name='conv_all')(merged_output)
    fc_full = Flatten(name="flatten")(conv_all)
    fc_full = Dense(3072, activation='relu', name="fc_full")(fc_full)

    #5 branches
    fc_detection  = Dense(512, activation='relu', name='fc_detection')(fc_full)
    fc_detection2 = Dense(2, activation='softmax', init='he_normal', name='fc_detection2')(fc_detection)

    fc_landmarks  = Dense(512, activation='relu', name='fc_landmarks')(fc_full)
    fc_landmarks2 = Dense(42, activation='softmax', init='he_normal', name='fc_landmarks2')(fc_landmarks)

    fc_visibility  = Dense(512, activation='relu', name='fc_visibility')(fc_full)
    fc_visibility2 = Dense(21, activation='softmax', init='he_normal', name='fc_visibility2')(fc_visibility)

    fc_pose  = Dense(512, activation='relu', name='fc_pose')(fc_full)
    fc_pose2 = Dense(3, activation='softmax', init='he_normal', name='fc_pose2')(fc_pose)

    fc_gender  = Dense(512, activation='relu', name='fc_gender')(fc_full)
    fc_gender2 = Dense(2, activation='softmax', init='he_normal', name='fc_gender2')(fc_gender)


    #Defining the model
    hyperface_model = Model(input=alex_model.input, output=[fc_detection2, fc_landmarks2, fc_visibility2, fc_pose2, fc_gender2])
    return hyperface_model


print get_hyperface_model("alexnet_weights.h5").summary()