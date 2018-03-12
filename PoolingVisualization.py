
# coding: utf-8

# In[1]:


# imports
from collections import Counter
import matplotlib as mpl
import h5py
from skimage import io
from matplotlib import gridspec
import matplotlib.collections
import matplotlib.patches as patches
from collections import Counter
import operator
import imageio
imageio.plugins.ffmpeg.download()
import os
import nibabel as nib
from nibabel.testing import data_path
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import time
from scipy import ndimage
from scipy.io import loadmat
import pickle
import sys
from sklearn.decomposition import PCA
import pylab
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import multiprocessing
import ipywidgets as widgets
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.fx.all import crop
from numpy import cross, eye, dot
from scipy.linalg import expm3, norm
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from scipy.stats.mstats import zscore
import timeit
get_ipython().magic('matplotlib inline')


# In[26]:


from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
import os
import keras
from keras.layers import Layer
from keras import metrics
from keras import backend as K

import sys

NX = 32
NY = 32
NC = 1
img_rows, img_cols, img_chns = NX, NY, NC

load_dir = os.path.join(os.getcwd(), '../../saved_models')

# dimensions of the generated pictures for each filter.
img_width = 32
img_height = 32

K.set_learning_phase(1)


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


bottleneck_mode = 'flexible'
first_layer_channels = 32
second_layer_stride = 1
#interface_nonlinearity = 'hard_sigmoid'
interface_nonlinearity = 'relu'
task = 'classification'
filter_size = 9
retina_layers = 2
brain_layers = 2



#one layer retina:
#model_name = 'cifar10_type_flexibleBigNLShft_noise_start_0.0_noise_end_1.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_shift_0.0_task_classification_filter_size_9_retina_layers_1_brain_layers2_batch_norm_0_bias_0'
#model_name = 'cifar10_type_flexibleBigNLShft_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_shift_0.0_task_classification_filter_size_9_retina_layers_1_brain_layers2_batch_norm_0_bias_0'


#model_name = 'cifar10_type_flexibleBigNLShft_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_1'


model_name = 'cifar10_type_flexibleBigNLReg_noise_start_0.0_noise_end_0.0_reg_1e-05_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_hard_sigmoid_task_classification_filter_size_9_retina_layers_2_brain_layers4_batch_norm_0'

#model_name = 'cifar10_type_flexibleBigNL_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_hard_sigmoid_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0'
#model_name = 'cifar10_type_test_noise_start_0.0_noise_end_1e-05_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0'
#model_name = 'cifar10_type_flexibleBigNL_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0'

model_path = os.path.join(load_dir, model_name)


model = keras.models.load_model(model_path)
#model_name = 'cifar10_type_flexibleBigNLShft_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_1'
#model_name = 'cifar10_type_flexibleBigNLShftFx_noise_start_0.0_noise_end_1.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_reluShift_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_1'

#model_name = 'cifar10_type_flexibleBigNLShftFx_noise_start_0.0_noise_end_1.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_reluShift_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0'
#model_name = 'cifar10_type_flexibleBigNLShftFx_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_reluShift_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0'

#reg (invreg below)
#model_name = 'cifar10_type_flexibleBigNLShftAR_NS_0.0_NE_1.0_reg_0.0_FC_32_SS_1_NL_relu_INL_relu_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0_actreg_4e-08_invreg_0'
#model_name = 'cifar10_type_flexibleBigNLShftAR_noise_start_0.0_noise_end_1.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0_actreg_1e-08_invreg_0'
#invreg
#model_name = 'cifar10_type_AR_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_relu_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0_actreg_10000000.0_invreg_1'

#invreg2 (anti-sparse)
#model_name = 'cifar10_type_flexibleBigNLShftAR_NS_0.0_NE_0.0_reg_0.0_FC_32_SS_1_NL_relu_INL_relu_shift_0.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0_actreg_1000000000.0_invreg_2'


#shift/bias
#model_name ='cifar10_type_flexibleBigNLShftFx_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_reluShift_shift_1.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0'
#model_name = 'cifar10_type_flexibleBigNLShft_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_reluShift_shift_1.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0'

#steep
#model_name = 'cifar10_type_flexibleBigNLShft_NS_0.0_NE_0.0_reg_0.0_FC_32_SS_1_NL_relu_INL_reluShallow_shift_256.0_task_classification_filter_size_9_retina_layers_2_brain_layers2_batch_norm_0_bias_0_actreg_0.0_invreg_0'

#caprelushallow
model_name = 'cifar10_type_flexibleBigNL_noise_start_0.0_noise_end_0.0_reg_0.0_first_channels_32_second_stride_1_nonlin_relu_interface_nonlin_reluShift_task_classification_filter_size_9_retina_layers_2_brain_layers4_batch_norm_0'


model_path = os.path.join(load_dir, model_name)
model.load_weights(model_path)
print('Model loaded.')

model.summary()


# In[59]:


F = np.zeros((2,50,32,32))
all_layers = ['conv2d_2']

for l in range(len(all_layers)):
    
    layer_name = all_layers[l]
    
    print(layer_name)
    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
    input_img = layer_dict['input_1'].output
    #print(layer_dict)


    kept_filters = []
    layer_output = layer_dict[layer_name].output
    #edge_conv = keras.layers.Conv2D(32, (32, 32), padding='same',  input_shape=input_img.shape, use_bias=False, kernel_initializer=keras.initializers.Constant(value=np.transpose(edge_filter, (1, 2, 0))))(input_img)
    #circular_conv = keras.layers.Conv2D(32, (32, 32), padding='same', input_shape=input_img.shape, use_bias=False, kernel_initializer=keras.initializers.Constant(value=np.transpose(circular_filter, (1, 2, 0))))(input_img)
   
    for filter_index in range(32):

        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        

        if K.image_data_format() == 'channels_first':
            image_rep_size_x = layer_output.shape[2]
            image_rep_size_y = layer_output.shape[3]
            loss = K.mean(layer_output[:, filter_index, image_rep_size_x//2, image_rep_size_y//2])
            
        else:
            image_rep_size_x = layer_output.shape[1]
            image_rep_size_y = layer_output.shape[2]
            loss = K.mean(layer_output[:, image_rep_size_x//2, image_rep_size_y//2, filter_index])


        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = 0*np.ones((1, 1, img_width, img_height)) #np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = 0*np.ones((1, img_width, img_height, 1)) #np.random.random((1, img_width, img_height, 3))
        #input_img_data = (input_img_data - 0.5) * 20 + 128


        # we run gradient ascent for 1 step so it's just a computation of the gradient
        for i in range(1):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
            print(np.linalg.norm(grads_value))


        # decode the resulting input image

        #img = deprocess_image(input_img_data[0])
        img = input_img_data[0]
        
        kept_filters.append((img, loss_value))
        F[l,filter_index,:,:] = np.squeeze(img)


    # we will stich the best 64 filters on a 8 x 8 grid.
    n = 5
    


    get_output = K.function([input_img], [layer_output])
    #get_edge_detect = K.function([input_img], [edge_conv])
    #get_circular_detect = K.function([input_img], [circular_conv])
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            try:
                img, loss = kept_filters[i * n + j]
                
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
            except:
                print('disaster')
                a = 1
            

    # save the result to disk
    #imsave('cifar10_type_'+bottleneck_mode+'_layer_name_'+layer_name+'_first_channels_'+str(first_layer_channels)+'_second_stride_'+str(second_layer_stride)+'_interface_nonlin_'+interface_nonlinearity+'_task_'+task+'_filter_size_'+str(filter_size)+'_retina_layers_'+str(retina_layers)+'_brain_layers'+str(brain_layers)+'.png', stitched_filters)
    


# In[58]:


import matplotlib.pyplot as plt
W,b = layer_dict['conv2d_4'].get_weights()



w = W[:,:,22,17]
w = w.reshape((9,9), order = 'F')


#w = w.reshape((9*9, 32))

print(W.shape,w.shape)

plt.figure(figsize = (20,5))
plt.imshow(w[:,:],cmap = 'seismic', aspect = 'auto', vmin = -0.2,vmax = 0.2)
#for i in range(31, 32):
#    plt.plot([9*i,9*i],[-0.5,8.5], 'k')
plt.xlabel('space x channel')
plt.ylabel('space')

plt.colorbar()
plt.show()



#b.shape


# In[56]:


W,b = layer_dict['conv2d_3'].get_weights()
print(b.shape)

w = W[:,:,:,6]
w = w.reshape((9,9*32), order = 'F')
print(W.shape,w.shape)

plt.figure(figsize = (20,5))
plt.imshow(w[:,:32*9],cmap = 'seismic', aspect = 'auto', vmin = -0.5,vmax = 0.5)
plt.colorbar()


# In[120]:


F = None


# In[60]:


for i in range(32):
    plt.figure()
    plt.imshow(F[0,i,:,:], cmap = 'seismic', vmin = -12, vmax = 12)
    plt.title(str(i))
    plt.colorbar()


# In[31]:


#plt.figure()
#plt.imshow(F[0,30,:,:], cmap = 'seismic', vmin = -12, vmax = 12)
#plt.colorbar()

#circular_filter = F[0, :, :, :]
#circular_filter1 = F[0, 3, :, :]
#circular_filter2 = F[0, 30, :, :]

for i in range(50):
    print(i)
    plt.imshow(edge_filter[i, :, :], cmap = 'seismic', vmin = -12, vmax = 12)
    plt.show()
    plt.imshow(circular_filter[i, :, :], cmap = 'seismic', vmin = -12, vmax = 12)
    plt.show()


# In[37]:


print(edge_filter1.shape)
print(edge_filter2.shape)
print(circular_filter1.shape)
print(circular_filter2.shape)
print(np.sum(np.square(edge_filter1)))
print(np.sum(np.square(edge_filter2)))
print(np.sum(np.square(circular_filter1)))
print(np.sum(np.square(circular_filter2)))


# In[13]:


from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.mean(x_train, 3, keepdims=True)
x_test = np.mean(x_test, 3, keepdims=True) 

print(x_train.shape)
edge_vals_specific = []
circular_vals_specific = []
for train_image in range(len(x_train)//20):
    if train_image % 100 == 0:
        print (train_image // 100)
    im = x_train[train_image, :, :, 0]

    for a in range(16, 48, 4):
        for b in range(16, 48, 4):
            imshift = np.zeros(shape=[96, 96])
            edge_filter_shift = np.zeros(shape=[96, 96])
            circular_filter_shift = np.zeros(shape=[96, 96])
            edge_filter_shift[32:64, 32:64] = edge_filter[27, :, :]
            circular_filter_shift[32:64, 32:64] = circular_filter[, :, :]
            imshift[a:(a+32), b:(b+32)] = im
            edge_like = np.sum(np.multiply(imshift, edge_filter_shift))
            circular_like = np.sum(np.multiply(imshift, circular_filter_shift))
            edge_vals_specific.append(edge_like)
            circular_vals_specific.append(circular_like)

plt.hist(edge_vals, bins=range(-20000, 20000, 200))
plt.show()
plt.hist(circular_vals, bins=range(-20000, 20000, 200))
plt.show()

plt.hist2d(edge_vals, circular_vals, bins=40, norm=LogNorm())
plt.colorbar()
plt.show()


plt.hist(edge_vals_specific, bins=range(-20000, 20000, 200))
plt.show()
plt.hist(circular_vals_specific, bins=range(-20000, 20000, 200))
plt.show()

plt.hist2d(edge_vals_specific, circular_vals_specific, bins=40, norm=LogNorm())
plt.colorbar()
plt.show()
    
    


# In[180]:


b = np.reshape(b, (1, 32, 1, 1))
print(b)


# In[14]:


from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from entropy_estimators import continuous


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.mean(x_train, 3, keepdims=True)
x_test = np.mean(x_test, 3, keepdims=True) 


arstcaprelu4layers = get_output([x_train[:1000]])[0]



    
#print(arst.shape)

'''
thing = np.reshape(x_train[100:200], (100, -1))

thingcirc = np.reshape(arstnobias[100:200], (100, -1))
thingedge = np.reshape(arstedgenobias[100:200], (100, -1))

thingcirc1layer = np.reshape(edgedetect[100:200], (100, -1))
thingedge1layer = np.reshape(circulardetect[100:200], (100, -1))

print(continuous.get_mi(thing,np.max(thingedge1layer, 0)))

print(continuous.get_mi(thing, np.max(thingcirc1layer, 0)))



print((-1 < arstshiftnobias).sum())
print((0 < arstedgenobias).sum())
print((0 < arstnobias).sum())

print(np.var(arstshiftnobias))
print(np.var(arstedgenobias))
print(np.var(arstnobias))

print(np.var(np.min(arstshiftnobias, -1)))
print(np.var(np.min(arstedgenobias, 0)))
print(np.var(np.min(arstnobias, 0)))

'''
'''
plt.hist(arstnobiasANTISPARSE[:, :, :, :].flatten(), bins=range(-10, 10, 1), normed=True)
plt.axvline(x=0, c='black')
plt.title('ANTISPARSE')
plt.show()
'''



plt.hist(arstcaprelu4layers[:, :, :, :].flatten(), bins=[x / 100.0 - 2.5 for x in range(1, 500)], normed=True)
plt.axvline(x=0, c='black')
plt.title('cap relu shallow 4 layers')
plt.show()

plt.hist(arsthardsig[:, :, :, :].flatten(), bins=[x / 100.0 for x in range(1, 100)], normed=True)
plt.axvline(x=0, c='black')
plt.title('hard sig')
plt.show()

plt.hist(arstonelayer[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
plt.axvline(x=0, c='black')
plt.title('Circular Filter Network ONE LAYER')
plt.show()
plt.hist(arstedgeonelayer[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
plt.axvline(x=0, c='black')
plt.title('Edge Filter Network ONE LAYER')
plt.show()

plt.hist2d(arstonelayer.flatten(), arstedgeonelayer.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.title('Joint Distribution, ONE LAYER RETINA activations')

plt.show()
    
plt.hist(arst[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
plt.axvline(x=0, c='black')
plt.title('Circular Filter Network with Bias, Second layer activations')
plt.show()
plt.hist(arstedge[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
plt.axvline(x=0, c='black')
plt.title('Edge Filter Network with Bias, Second layer activations')
plt.show()

plt.hist2d(arst.flatten(), arstedge.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.title('Joint Distribution, Second layer activations (with bias)')

plt.show()


plt.hist(arstnobiasREG[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
#plt.axvline(x=0, c='black')
plt.ylim(0, 0.004)
plt.title('Circular Filter Network with REG, Second layer activations')
plt.show()

plt.hist(arstnobiasREGLESS[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
#plt.axvline(x=0, c='black')
plt.ylim(0, 0.004)
plt.title('Circular Filter Network with REG (less), Second layer activations')
plt.show()



'''
plt.hist(arstshiftnobias[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
#plt.axvline(x=0, c='black')
plt.title('Shift 1.0 Network no bias, Second layer activations')
plt.show()



plt.hist(arstedgenobiasINVREG[:, :, :, :].flatten(), bins=range(-1000, 10000, 10), normed=True)
#plt.axvline(x=0, c='black')
plt.ylim(0, 0.0025)
plt.title('Edge Filter Network with INV REG, Second layer activations')
plt.show()
'''

plt.hist(arstnobias[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
plt.axvline(x=0, c='black')
plt.title('Circular Filter Network, Second layer activations')
plt.show()
plt.hist(arstedgenobias[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
plt.axvline(x=0, c='black')
plt.title('Edge Filter Network, Second layer activations')


plt.show()

plt.hist(arstnobiaslayer1[:, :, :, :].flatten(), bins=range(-200, 200, 2), normed=True)
plt.axvline(x=0, c='black')
plt.title('Circular Filter Network, First layer activations')

plt.show()
plt.hist(arstedgenobiaslayer1[:, :, :, :].flatten(), bins=range(-200, 200, 2), normed=True)
plt.axvline(x=0, c='black')
plt.title('Edge Filter Network, First layer activations')


plt.show()

plt.hist2d(arstnobias.flatten(), arstedgenobias.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.title('Joint Distribution, Second layer activations')

plt.show()

plt.hist2d(arstnobiaslayer1.flatten(), arstedgenobiaslayer1.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.title('Joint Distribution, First layer activations')

plt.show()


# In[28]:


plt.hist2d(arstedgenobias.flatten(), arstedge.flatten(), bins=40, norm=LogNorm())
plt.colorbar()
plt.show()


# In[31]:


np.save('arst.npy', arst)
np.save('arstedge.npy', arstedge)
np.save('arstonelayer.npy', arstonelayer)
np.save('arstedgeonelayer.npy', arstedgeonelayer)
np.save('arstnobias.npy', arstnobias)
np.save('arstedgenobias.npy', arstedgenobias)
np.save('arstnobiaslayer1.npy', arstnobiaslayer1)
np.save('arstedgenobiaslayer1.npy', arstedgenobiaslayer1)
np.save('circulardetect.npy', circulardetect)
np.save('edgedetect.npy', edgedetect)
np.save('arstshiftnobias.npy', arstshiftnobias)
np.save('arstshiftnobias.npy', arstshiftnobias)
np.save('arstnobiasREG.npy', arstnobiasREG)
np.save('arstnobiasREGLESS.npy', arstnobiasREGLESS)



# In[2]:


import numpy as np
arst = np.load('arst.npy')
arstedge = np.load('arstedge.npy')
arstonelayer = np.load('arstonelayer.npy')
arstedgeonelayer = np.load('arstedgeonelayer.npy')
arstnobias = np.load('arstnobias.npy')
arstedgenobias = np.load('arstedgenobias.npy')
arstnobiaslayer1 = np.load('arstnobiaslayer1.npy')
arstedgenobiaslayer1 = np.load('arstedgenobiaslayer1.npy')
circulardetect = np.load('circulardetect.npy')
edgedetect = np.load('edgedetect.npy')
arstshiftnobias = np.load('arstshiftnobias.npy')
arstnobiasREG = np.load('arstnobiasREG.npy')
arstnobiasREGLESS = np.load('arstnobiasREGLESS.npy')



# In[3]:


edge_filter = edge_filter[:32]
circular_filter = circular_filter[:32]


# In[30]:


from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.mean(x_train, 3, keepdims=True)
x_test = np.mean(x_test, 3, keepdims=True) 


#edgedetect = get_edge_detect([x_train[:1000]])[0]
#circulardetect = get_circular_detect([x_train[:1000]])[0]

print(arst.shape)

print(np.mean(np.abs(edgedetect)))
print(np.mean(np.abs(circulardetect)))

edgedetectnorm = 100 * edgedetect / (np.mean(np.abs(edgedetect)))
circulardetectnorm = 100 * circulardetect / (np.mean(np.abs(circulardetect)))

print(np.var(edgedetectnorm))
print(np.var(circulardetectnorm))

                

plt.hist(edgedetectnorm[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
plt.ylim(0, 0.01)
#plt.axvline(x=0, c='black')
plt.title('Direct Projection onto edge receptive fields')
plt.show()
plt.hist(circulardetectnorm[:, :, :, :].flatten(), bins=range(-1000, 1000, 10), normed=True)
#plt.axvline(x=0, c='black')
plt.ylim(0, 0.01)

plt.title('Direct Projection onto circular receptive fields')

plt.show()


plt.hist2d(edgedetect.flatten(), circulardetect.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.title('Joint Distribution of Direct Projections onto Receptive fields')

plt.colorbar()
plt.show()



# In[86]:


from scipy.stats import pearsonr
plt.hist2d(edgedetect.flatten(), arstedgenobias.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.show()
print(pearsonr(edgedetect.flatten(), arstedgenobias.flatten()))

plt.hist2d(circulardetect.flatten(), arstnobias.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.show()
print(pearsonr(circulardetect.flatten(), arstnobias.flatten()))


# In[100]:


from scipy.stats import pearsonr
plt.hist2d(edgedetect.flatten(), arstnobiasedgelayer1.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.show()
print(pearsonr(edgedetect.flatten(), arstedgenobiaslayer1.flatten()))

plt.hist2d(circulardetect.flatten(), arstnobiaslayer1.flatten(), bins=40, norm=LogNorm(), normed=True)
plt.colorbar()
plt.show()
print(pearsonr(circulardetect.flatten(), arstnobiaslayer1.flatten()))


# In[54]:


edgevararray = np.array(edgevarlist)
circularvararray = np.array(circularvarlist)
print(edgevararray.shape)
print(circularvararray.shape)
print(len(x_train_augs))
print(x_train_augs[0].shape)


# In[ ]:


from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.mean(x_train, 3, keepdims=True)
x_test = np.mean(x_test, 3, keepdims=True) 


