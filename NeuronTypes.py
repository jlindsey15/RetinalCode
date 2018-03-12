
# coding: utf-8

# In[9]:

'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

Results example: http://i.imgur.com/4nj4KjN.jpg
'''

import numpy as np
import scipy.ndimage
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def pad_with(vector, pad_width, iaxis, kwargs):
     pad_value = kwargs.get('padder', 10)
     vector[:pad_width[0]] = pad_value
     vector[-pad_width[1]:] = pad_value
     return vector


data = np.load('RGCoutCT.npy')
print(data.shape)
data = np.reshape(data, [-1, 10000])
datanorm = np.linalg.norm(data, axis=0)

pca = PCA(n_components=50)
pca.fit(data)
snapshots_pca = pca.transform(data)
print(snapshots_pca.shape)
print(pca.explained_variance_ratio_)
tsne = TSNE(n_components=2)
snapshots_pca_tsne = tsne.fit_transform(snapshots_pca)
print(snapshots_pca_tsne.shape)

kmeans = KMeans(n_clusters=2, random_state=0).fit(snapshots_pca_tsne)

xindices = []
yindices = []

labels = np.reshape(kmeans.labels_, [-1])
np.random.shuffle(labels)

labels = np.reshape(labels, [16, 16, 4])



for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if np.sum(labels[i, j, :]) > 0 and np.sum(labels[i, j, :]) < 4:
            xindices.append(i)
            yindices.append(j)

print(len(xindices) / (16*16))

xs = np.array(xindices)
ys = np.array(yindices)


#plt.hist2d(xindices, yindices)
#plt.colorbar()
plt.scatter(xindices, yindices)
plt.show()

'''

xindices = []
yindices = []
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if np.sum(labels[i, j, :]) == 0:
            xindices.append(i)
            yindices.append(j)

xs = np.array(xindices)
ys = np.array(yindices)

#labels = 5 + (10*labels)

#plt.scatter(snapshots_pca_tsne[:, 0], snapshots_pca_tsne[:, 1], c=np.reshape(labels, [-1]))
plt.scatter(xindices, yindices)
plt.show()
'''
