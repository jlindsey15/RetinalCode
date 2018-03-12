# RetinalCode

Note: The code present on this repo is currently poorly documented.  In the coming weeks this will be addressed, and instructions will be given for how to reproduce the results shown in the figures.

Identifying the constraints and redundancies shaping the retinal code with a deep network simulation.  We find that convolutional and (untied) locally connected networks tend to yield oriented receptive fields similar to those in V1.  However, subject to a dimensionality bottleneck constraint in an early layer, center-surround receptive fields characteristic of retinal ganglion cells and LGN emerge. 


Below: A selection of oriented receptive fields in the second layer of the control network. B:
center-surround receptive field in the second (bottleneck) layer of a bottlenecked network. C: A selection
of oriented receptive fields in the third layer of the bottlenecked network.

![Alt text](figures/RFs.png?raw=true "Receptive Fields")


Moreover, given enough neurons in the bottleneck layer, clear and distinct cell types emerge.  This occurs even when the parameter-sharing constraint present in convolutional networks is removed.

Below: Clustering of cell responses according t-SNE.  These correspond to ON and OFF center-surround receptive fields.

![Alt text](figures/CellTypes.png?raw=true "Cell Types")

Downstream of the bottleneck, oriented receptive fields re-appear, pooling from center-surround neurons in the previous layer as in Hubel and Wiesel's hypothesis.

Below: A representative example of an orientation-selective neuron (receptive field at right) drawing
from center-surround channels (receptive fields at left) in the previous layer with weight matrices
(center) according to their polarity. Light / dark-selective regions of a receptive field, and positive /
negative weights, are represented with red / blue, respectively.

![Alt text](figures/Pooling.png?raw=true "Pooling Visualization")

We can also obtain localized center-surround receptive fields when we train a one-layer locally connected bottlenecked autoencoder on the activations of the orientation-selective neurons in the original network architecture.  This allows us to test what properties of the activations of these neurons makes them well-encoded by center-surround neurons.  We test this by training the autoencoder while enforcing artificial statistics on the activations of the orientation-selective neurons and viewing the resulting receptive fields of the encoding neurons.  Evidence suggests that correlated activity between neurons of different types and at different spatial locations is important for making center-surround an effective compressed encoding.

Below: Receptive fields of neurons trained to encode artificially generated activations of orientation-selective neuron with the following distributions. A: The true distribution of orientationselective neuron activations in response to natural scenes. B: Multivariate gaussian with true mean and covariance. C: Multivariate Gaussian with true mean and covariance between neurons across channels at the same spatial location. D: Multivariate Gaussian with true mean and covariance between neurons across locations in the same convolutional channel. E: Gaussian with true mean and variance of each neuron. F: Standard random normal.

![Alt text](figures/CovarExperiments.png?raw=true "ArtificialStatistics")



