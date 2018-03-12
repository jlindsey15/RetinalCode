
# coding: utf-8

# In[2]:




# In[11]:

'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Layer, BatchNormalization, LocallyConnected2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, GaussianNoise, UpSampling2D, Input
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras import metrics
from keras.models import Model

import numpy as np
import sys
import os


bottleneck_mode = sys.argv[1]
noise_start = float(sys.argv[2])
noise_end = float(sys.argv[3])
reg = float(sys.argv[4])
first_layer_channels = int(sys.argv[5])
second_layer_stride = int(sys.argv[6])
nonlinearity = sys.argv[7]
interface_nonlinearity = sys.argv[8]
shft = float(sys.argv[9])
task = sys.argv[10]
filter_size = int(sys.argv[11])
retina_layers = int(sys.argv[12])
brain_layers = int(sys.argv[13])
use_b = int(sys.argv[14])
actreg = float(sys.argv[15])
invreg = int(sys.argv[16])
#batch_norm = int(sys.argv[12])
batch_norm = 0
batch_size = 8
num_classes = 10
epochs = 4
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_type_'+bottleneck_mode+'_NS_'+str(noise_start)+'_NE_'+str(noise_end)+'_reg_'+str(reg)+'_FC_'+str(first_layer_channels)+'_SS_'+str(second_layer_stride)+'_NL_'+nonlinearity+'_INL_'+interface_nonlinearity+'_shift_'+str(shft)+'_task_'+task+'_filter_size_'+str(filter_size)+'_retina_layers_'+str(retina_layers)+'_brain_layers'+str(brain_layers)+'_batch_norm_'+str(batch_norm)+'_bias_'+str(use_b)+'_actreg_'+str(actreg)+'_invreg_'+str(invreg)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

if use_b == 1:
    use_b = True
else:
    use_b = False
    



def create_relu_advanced(max_value=1., shallow=1.0, shift=0):        
    def relu_advanced(x):
        if max_value == None:
            mv = None
        else:
            mv = K.cast_to_floatx(max_value)
        return K.relu(shallow*(x+shift), max_value=mv)
    return relu_advanced

if interface_nonlinearity == 'capRelu':
    interface_nonlinearity = create_relu_advanced(max_value=1)
if nonlinearity == 'capRelu':
    nonlinearity = create_relu_advanced(max_value=1)
    
if interface_nonlinearity == 'capReluShallow':
    interface_nonlinearity = create_relu_advanced(max_value=1, shallow=0.2)
if nonlinearity == 'capReluShallow':
    nonlinearity = create_relu_advanced(max_value=1, shallow=0.2)
    
if interface_nonlinearity == 'capReluShift':
    interface_nonlinearity = create_relu_advanced(max_value=1, shift=0.5)
if nonlinearity == 'capReluShift':
    nonlinearity = create_relu_advanced(max_value=1, shift = 2.5)
    
if interface_nonlinearity == 'capReluShiftShallow':
    interface_nonlinearity = create_relu_advanced(max_value=1, shallow=0.2, shift=2.5)
if nonlinearity == 'capReluShiftShallow':
    nonlinearity = create_relu_advanced(max_value=1, shallow=0.2, shift=2.5)
    
if interface_nonlinearity == 'reluShiftShallow':
    interface_nonlinearity = create_relu_advanced(max_value=None, shallow=0.2, shift=2.5)
if nonlinearity == 'reluShiftShallow':
    nonlinearity = create_relu_advanced(max_value=None, shallow=0.2, shift=2.5)
    
if interface_nonlinearity == 'reluShift':
    interface_nonlinearity = create_relu_advanced(max_value=None,  shift=shft)
if nonlinearity == 'reluShift':
    nonlinearity = create_relu_advanced(max_value=None,  shift=shft)
    
if interface_nonlinearity == 'reluShallow':
    interface_nonlinearity = create_relu_advanced(max_value=None,  shallow=shft)
if nonlinearity == 'reluShallow':
    nonlinearity = create_relu_advanced(max_value=None,  shallow=shft)
    
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.mean(x_train, 3, keepdims=True)
x_test = np.mean(x_test, 3, keepdims=True) 
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#model.add(UpSampling2D(size=(1, 1), data_format=None, input_shape=x_train.shape[1:]))
model.add(GaussianNoise(noise_start, input_shape=x_train.shape[1:]))
'''
if bottleneck_mode == 'append_retina':
    #RETINA net

    model.add(Conv2D(2, (5, 5), strides=(5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('tanh'))
    model.add(Conv2D(30, (5, 5), strides=(11, 11)))
    model.add(Activation('tanh'))
    model.add(GaussianNoise(noise_end))
          
    #INVERSE RETINA net
    model.add(Conv2DTranspose(30, (5, 5), strides=(11, 11)))
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(30, (5, 5), strides=(5, 5)))
    model.add(Activation('tanh'))


if bottleneck_mode == 'append_control':
    #'RETINA' net no bottleneck

    model.add(Conv2D(2, (5, 5), strides=(1, 1), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('tanh'))
    model.add(Conv2D(30, (5, 5), strides=(1, 1)))
    model.add(Activation('tanh'))
    model.add(GaussianNoise(noise_end))
          
    #INVERSE 'RETINA' net no bottleneck
    model.add(Conv2DTranspose(30, (5, 5), strides=(1, 1)))
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(30, (5, 5), strides=(1, 1)))
    model.add(Activation('tanh'))

 '''    

def create_inverse_l1_reg(ar):
    def inverse_l1_reg(weight_matrix):
        return ar / K.sum(K.abs(weight_matrix))
    return inverse_l1_reg

def create_nonsparse_l1_reg(ar):
    def nonsparse_l1_reg(weight_matrix):
        result = ar * K.sum(K.relu(-weight_matrix))
        return result
    return nonsparse_l1_reg

inverse_l1_reg = create_inverse_l1_reg(actreg)
nonsparse_l1_reg = create_nonsparse_l1_reg(actreg)

if invreg == 2:
    intreg = nonsparse_l1_reg
elif invreg == 1:
    intreg = inverse_l1_reg
else:
    intreg = keras.regularizers.l1(actreg)

filters = 64
NX = 32
NY = 32
NC = 1
img_rows, img_cols, img_chns = NX, NY, NC
num_conv = 3
latent_dim = 10
intermediate_dim = 1024
x = Input(shape=x_train[0].shape)
gn = GaussianNoise(0)(x)
#gn = Flatten()(gn)
if retina_layers > 2:
    conv1 = Conv2D(first_layer_channels, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg),  padding='same', input_shape=x_train.shape[1:])(x)
    conv1_relu = Activation(nonlinearity)(conv1)
    conv2 = Conv2D(32, (filter_size, filter_size), strides=(second_layer_stride,second_layer_stride), kernel_regularizer=keras.regularizers.l1(reg), padding='same')(conv1_relu)
    conv2_nonlin = Activation(nonlinearity)(conv2)
    for iterationX in range(retina_layers - 2):
        if iterationX == retina_layers - 3:
            conv2 = Conv2D(32, (filter_size, filter_size), strides=(second_layer_stride,second_layer_stride), kernel_regularizer=keras.regularizers.l1(reg), padding='same', use_bias=use_b)(conv2_nonlin)
            conv2_nonlin = Activation(interface_nonlinearity)(conv2)
        else:
            conv2 = Conv2D(32, (filter_size, filter_size), strides=(second_layer_stride,second_layer_stride), kernel_regularizer=keras.regularizers.l1(reg), padding='same')(conv2_nonlin)
            conv2_nonlin = Activation(nonlinearity)(conv2)
if retina_layers == 2:
    conv1 = Conv2D(32, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', input_shape=x_train.shape[1:], name='conv1', trainable=False)(gn) #conv1 = Dense(8*32*32, kernel_regularizer=keras.regularizers.l1(1.0), activity_regularizer=keras.regularizers.l1(actreg), name='crazy1' )(gn) 
    conv1_relu = Activation(nonlinearity)(conv1)
    #conv1_relu = GaussianNoise(0.1)(conv1_relu)
    #conv1_relu = keras.layers.Reshape([32, 32, 8])(conv1_relu)
    
    #conv1_relu = GaussianNoise(noise_start)(conv1_relu)
    #conv2 = Dense(8*32*32, kernel_regularizer=keras.regularizers.l1(reg), activity_regularizer=keras.regularizers.l1(actreg), name='crazy2')(conv1_relu)
    conv2 = Conv2D(32, (filter_size, filter_size), strides=(second_layer_stride,second_layer_stride), kernel_regularizer=keras.regularizers.l1(reg), padding='same',  activity_regularizer=intreg, use_bias=use_b, name='conv2', trainable=False)(conv1_relu)
    #conv2 = BatchNormalization()(conv2)
    conv2_nonlin = Activation(nonlinearity)(conv2)
    gn = ZeroPadding2D((4, 4))(gn)
    crazyconv1 = LocallyConnected2D(150, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='valid', input_shape=x_train.shape[1:], name='crazyconv1', trainable=True)(gn) #conv1 = Dense(8*32*32, kernel_regularizer=keras.regularizers.l1(1.0), activity_regularizer=keras.regularizers.l1(actreg), name='crazy1' )(gn) 
    print('cc1', crazyconv1.shape)
    crazyconv1_relu = Activation(nonlinearity)(crazyconv1)
    #conv1_relu = GaussianNoise(0.1)(conv1_relu)
    #conv1_relu = keras.layers.Reshape([32, 32, 8])(conv1_relu)
    
    #crazyconv1_relu = GaussianNoise(0.1)(crazyconv1_relu)
    crazyconv1_relu = ZeroPadding2D((4, 4))(crazyconv1_relu)

    #conv2 = Dense(8*32*32, kernel_regularizer=keras.regularizers.l1(reg), activity_regularizer=keras.regularizers.l1(actreg), name='crazy2')(conv1_relu)
    crazyconv2 = LocallyConnected2D(1, (filter_size, filter_size), strides=(second_layer_stride,second_layer_stride), kernel_regularizer=keras.regularizers.l1(reg), padding='valid',  activity_regularizer=intreg, use_bias=use_b, name='crazyconv2', trainable=True)(crazyconv1_relu)
    print('cc2', crazyconv2.shape)
    crazyconv2_nonlin = Activation(nonlinearity)(crazyconv2)
    crazyconv2_nonlin = ZeroPadding2D((4, 4))(crazyconv2_nonlin)
    
    
    #conv2_nonlin = GaussianNoise(0.1)(conv2_nonlin)
    #conv2_nonlin = Flatten()(conv2_nonlin)
    #conv2_nonlin = keras.regularizers.ActivityRegularizer(l1=actreg)
elif retina_layers == 1:
    conv2_nonlin = Conv2D(first_layer_channels, (filter_size, filter_size), strides=(second_layer_stride,second_layer_stride), kernel_regularizer=keras.regularizers.l1(reg), padding='same', use_bias=use_b,  activity_regularizer=intreg, input_shape=x_train.shape[1:])(x)
    conv2_nonlin = Activation(interface_nonlinearity)(conv2)
    #activity_regularizer=intreg
    

    
if batch_norm == 1:
    conv2_nonlin = BatchNormalization()(conv2_nonlin)
if noise_end > 0:
    conv2_nonlin = GaussianNoise(noise_end)(conv2_nonlin)

#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#I know this code is ridiculously ugly but I'd rather ensure consistency with past versions than risk messing it up

if brain_layers > 2:
    conv3 = Conv2D(64, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same')(conv2_nonlin)
    conv3_relu = Activation(nonlinearity)(conv3)
    conv4 = Conv2D(64, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same')(conv3_relu)
    conv4_relu = Activation(nonlinearity)(conv4)
    for iterationX in range(brain_layers - 2):
        conv4 = Conv2D(64, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same')(conv4_relu)
        conv4_relu = Activation(nonlinearity)(conv4)
    flattened = Flatten()(conv4_relu)
if brain_layers == 2:
    #conv3 = Dense(intermediate_dim, kernel_regularizer=keras.regularizers.l1(reg))(conv2_nonlin)
    
    print('baaa')
    print(conv2_nonlin.shape)
    conv3 = Conv2D(64, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='conv3', trainable=False)(conv2_nonlin)
    conv3_relu = Activation(nonlinearity)(conv3)
    outyou = conv2_nonlin
    #outyou = BatchNormalization()(conv2_nonlin)
    
    crazyconv3 = LocallyConnected2D(32, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='valid', name='crazyconv3', trainable=True)(crazyconv2_nonlin)
    print('cc3', crazyconv3.shape)
    #crazyconv2_nonlin = Activation(nonlinearity)(crazyconv2)
    crazyconv3_relu = Activation(nonlinearity)(crazyconv3)
    outme = crazyconv3_relu
    #conv4 = Dense(intermediate_dim, kernel_regularizer=keras.regularizers.l1(reg))(conv3_relu)
    conv4 = Conv2D(64, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='conv4', trainable=False)(conv3_relu)
    conv4_relu = Activation(nonlinearity)(conv4)
    flattened = Flatten()(conv4_relu)
elif brain_layers == 1:
    conv4 = Conv2D(64, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same')(conv2_nonlin)
    conv4_relu = Activation(nonlinearity)(conv4)
    flattened = Flatten()(conv4_relu)
elif brain_layers == 0:
    flattened = conv2_nonlin#Flatten()(conv2_nonlin)
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = K.sum(K.square(x-x_decoded_mean_squash))
        #xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        #kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


diff = CustomVariationalLayer()([outme, outyou])


hidden = Dense(intermediate_dim, kernel_regularizer=keras.regularizers.l1(reg), activity_regularizer=keras.regularizers.l1(actreg), name='dense1', trainable=False)(flattened)
hidden = Activation(nonlinearity)(hidden)
#hidden = GaussianNoise(noise_start)(hidden)
#model.add(Dropout(0.5))
pre_output = Dense(num_classes, name='dense2', trainable=False)(hidden)
output = Activation('softmax')(pre_output)

'''
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = z_mean#Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation=nonlinearity)
decoder_upsample = Dense(int(filters * NX/2 * NY/2), activation=nonlinearity)

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, int(NX/2), int(NY/2))
else:
    output_shape = (batch_size, int(NX/2), int(NY/2), filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation=nonlinearity)
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation=nonlinearity)
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation=nonlinearity)
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')


hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
if brain_layers > 2:
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    for iterationX in range(brain_layers - 2):
        deconv_2_decoded = decoder_deconv_2(deconv_2_decoded)
if brain_layers == 2:
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
elif brain_layers == 1:
    deconv_2_decoded = decoder_deconv_2(reshape_decoded)
elif brain_layers == 0:
    deconv_2_decoded = reshape_decoded
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        #kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


y = CustomVariationalLayer()([x, x_decoded_mean_squash])
#print(y.shape)
'''


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

if task == 'classification':
    model = Model(x, diff)
    model.load_weights(model_path, by_name=True)

    # Let's train the model using RMSprop
    #model.compile(loss='categorical_crossentropy',
    #              optimizer=opt,
    #               metrics=['accuracy'])
    model.compile(optimizer=opt, loss=None)

elif task == 'reconstruction':
    model = Model(x, y)

    model.compile(optimizer=opt, loss=None)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    if task == 'classification':
        model.fit(x_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, None),
                  shuffle=True)
    elif task == 'reconstruction':
        model.fit(x_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test,  None),
                  shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    if task == 'classification':
        model.fit_generator(datagen.flow(x_train, x_train,
                                         batch_size=batch_size),
                            steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                            epochs=epochs,
                            validation_data=(x_test, None),
                            workers=4)
    elif task == 'reconstruction':
        model.fit_generator(datagen.flow(x_train, x_train,
                                         batch_size=batch_size),
                            steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                            epochs=epochs,
                            validation_data=(x_test, None),
                            workers=4)        

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = 'NMbnlc150'+model_name
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
'''
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
'''







# In[ ]:


