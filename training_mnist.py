import numpy as np
import keras
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Convolution2D, concatenate
from quaternion_layers.dense import QuaternionDense
from quaternion_layers.conv import QuaternionConv2D
from quaternion_layers.bn import QuaternionBatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from numpy import linalg as LA
import tensorflow as tf

batch_size = 500
num_classes = 10
epochs = 10
num_perturb = 10
outputFilename = 'stabilityResults.txt'

def learnVectorBlock(I):
    """Learn initial vector component for input."""

    O = Convolution2D(1, (5, 5),
                      padding='same', activation='relu')(I)

    return O

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

datagen.fit(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

R = Input(shape=input_shape)

I = learnVectorBlock(R)
J = learnVectorBlock(R)
K = learnVectorBlock(R)

O = concatenate([R, I, J, K], axis=-1)

O = QuaternionConv2D(32, (5, 5), activation='relu', padding="same", kernel_initializer='quaternion', perturb = True)(O)
O = QuaternionConv2D(64, (5, 5), activation='relu', padding="same", kernel_initializer='quaternion')(O)
O = QuaternionConv2D(32, (5, 5), activation='relu', padding="same", kernel_initializer='quaternion')(O)

# O = Convolution2D(256, (5, 5), activation='relu', padding="same")(R)
# O = Convolution2D(256, (5, 5), activation='relu', padding="same")(O)
# O = Convolution2D(128, (5, 5), activation='relu', padding="same")(O)

O = Flatten()(O)
O = QuaternionDense(82, activation='relu', kernel_initializer='quaternion')(O)
O = QuaternionDense(48, activation='relu', kernel_initializer='quaternion')(O)
#O = Dropout(0.5)(O)
O = Dense(num_classes, activation='softmax')(O)

model = Model(R, O)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                           steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                           validation_data=(x_test, y_test))
np.save('mnist_results.npy', hist.history['val_accuracy'])

# layers = ['quaternion_conv2d','quaternion_conv2d_2','flatten','dense']
layers = ['quaternion_conv2d','dense']
layer_outputs = []

for layer in layers:
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer).output)
    layer_outputs.append(intermediate_layer_model.predict(x_test))

origWeights = np.copy(model.get_layer('quaternion_conv2d').get_weights()[0])
origBias = np.copy(model.get_layer('quaternion_conv2d').get_weights()[1])
alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
with open(outputFilename, "w") as of:
    for alpha in alphas:

        loss = np.zeros(num_perturb)
        acc = np.zeros(num_perturb)
        firstConvNorm = np.zeros(num_perturb)
        lastConvNorm = np.zeros(num_perturb)
        flattenNorm = np.zeros(num_perturb)
        outputNorm = np.zeros(num_perturb)

        for i in range(num_perturb):
            print(i)
            # perturbedWeights
            perturbedWeights = np.copy(origWeights)
            perturbedWeights[:,:,:,:64] += tf.random.uniform(perturbedWeights[:,:,:,:64].shape, minval=-alpha, maxval=alpha, seed=i)
            model.get_layer('quaternion_conv2d').set_weights([perturbedWeights, origBias])
            l, a = model.evaluate(x = x_test, y = y_test, verbose=0, batch_size=batch_size)
            perturbed_layer_outputs = []
            for layer in layers:
                perturbed_layer_model = Model(inputs=model.input,
                                        outputs=model.get_layer(layer).output)
                perturbed_layer_outputs.append(perturbed_layer_model.predict(x_test))


            loss[i] = l
            acc[i] = a
            firstConvNorm[i] = LA.norm(layer_outputs[0] - perturbed_layer_outputs[0])
            #lastConvNorm[i] = LA.norm(layer_outputs[1] - perturbed_layer_outputs[1])
            #flattenNorm[i] = LA.norm(layer_outputs[2] - perturbed_layer_outputs[2])
            outputNorm[i] = LA.norm(layer_outputs[1] - perturbed_layer_outputs[1])
        
        print(alpha)
        of.write(str(alpha) + '\n')

        lossStr = f'Loss: {np.mean(loss), np.std(loss)}\n'
        print(lossStr)
        of.write(lossStr)

        accStr = f'Acc: {np.mean(acc), np.std(acc)}\n'
        print(accStr)
        of.write(accStr)

        fConvNormStr = f'FirstConvNorm: {np.mean(firstConvNorm), np.std(firstConvNorm)}\n'
        print(fConvNormStr)
        of.write(fConvNormStr)

        oNormStr = f'OutputNorm: {np.mean(outputNorm), np.std(outputNorm)}\n'
        print(oNormStr)
        of.write(oNormStr)
        
        # print(f'LastConvNorm: {np.mean(lastConvNorm), np.std(lastConvNorm)}')
        # print(f'FlattenNorm: {np.mean(flattenNorm), np.std(flattenNorm)}')
        print()
        of.write('\n')