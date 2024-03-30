13) What is Flatten? Where does it need to be included in the MNIST CNN? Why?
- Flatten is layer to convert a matrix like output to a vector. And we need to do that
to perform classification with a fully connected network.

16) What is a "receptive field"?
- The area (group of pixels) in an image used to compute the output features in layer of a CNN, the deeper we are in the network (specifically, the more stride-2 convs we have before a layer), the larger the receptive field for an activation in that layer.

23) Why do we double the number of filters after each stride-2 conv?
- In order to keep the same amount of computation, this is, when we use stride-2 we downside to half the number of features to use, and we don't want to reduce the computation when we go deeper in the CNN, because we need to capture richer features.

28) Why are activations near zero problematic?
- Because we are multiplying by zero, which means the params aren't going work at all.

29) What are the upsides and downsides of training with a larger batch size?
- Upsides: More stable (representative) values for gradients, better use of GPUs or TPUs. Downsides: fewer batches per epoch, will means less opportunities to update the weights (can affect generalization), sometimes it can fit a batch in a GPU (but there is some workarounds...don't buy expensive chips).

30) Why should we avoid using a high learning rate at the start of training?
- To avoid inestable training (optimization), we may be jumping from different zones without skip over a minimum.

31) What is 1cycle training?
- A combination of two approaches: We should change the learning rate during training, from low, to high, and then back to low again.

40) Why do models with batch normalization layers generalize better?
- Since each layer don't be influenced too much for previous layers, so, for instance the activations near-zero are reduced.
- [ChatGPT] Machine Learning models with batch normalization layers tend to generalize better because batch normalization stabilizes the learning process by normalizing the input to a layer for each mini-batch. This ensures that the distribution of inputs to a layer remains more consistent during training. As a result, it reduces the internal covariate shift, which is the change in the distribution of network activations due to the update of weights. This stabilization allows for higher learning rates and less sensitivity to initialization, accelerating the training process. Additionally, batch normalization acts as a form of regularization, slightly smoothing the optimization landscape and introducing noise to the training process, which can help to prevent overfitting on the training data.