- 3) Explain how the "pixel similarity" approach to classifying digits works.

First we get a representative image for each class, how to? computing average pixel values between pixels at the same positions considering all the images.
Then we calculate a distance for a given input, for example mean absolute value, to each class, the minimum distance is the class predicted. Because more pixels roughly "matched" its values and then thaht reduces the "distance".

- 5) What is a "rank-3 tensor"?

A tensor with three dimensions.

- 6) What is the difference between tensor rank and shape? How do you get the rank from the shape?

The shape is the size for each dimension, and the rank is the number of dimensions.

- 7) What are RMSE and L1 norm?

RMSE is root mean squared error, basically, it is the mean square root of the difference to the power 2 between a collections of di-tuples. The L1 norm is the absolute mean value.

- 8) How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

Using broadcasting, with Numpy or PyTorch.

- 10) What is broadcasting?

Perform an operation between 2 tensors with different ranks, where a tensor is simulated to be "expanded" in order to compute efficiently the operation. In a nuthsell, refers to the ability to perform element-wise operations on arrays of different shapes and size, whithout explicitly reshaping them.

- 13) What is SGD? Why does SGD use mini-batches?

Stochastic Gradient Descent, means a technique to optimize the loss functions through updating the params of the models. It uses mini-batches because is the stable option between updating for each input or after forwarding a whole epoch.

- 14) What are the seven steps in SGD for machine learning?
    - Initialize weights (params in generall)
    - Forward a mini-batch.
    - Compute the mean loss.
    - Compute the gradients.
    - Update the weights using the gradients, backward.
    - Repeat that for each mini-batch until complete an epoch.
    - Repeat many epochs as needed.

- 15) Why can't we always use a high learning rate?

Because the optimization process could be very inestable, where happens "large" jumps.

- 18) What is a "gradient"?

It's a vector, containing the derivatives values of a composed function. In the context of deep learning, indicates how much change a param in order to reduce the loss function.

- 20) Why can't we use accuracy as a loss function?

Becuse it's a human metric that doesn't work well to drive the optimizacion process through derivatives (gradient descent). The accuracy metric is a non-smooth functions, which indicates that the gradient would be 0 almost all the time.

- 21) What is the difference between a loss function and a metric?

The loss is intent to help the optimization process, a metric is more intuitive to humans to know how better is the model (it could be accuract, F1, precision, recall).

- 23) What is the function to calculate new weights using a learning rate?

```python
weights = weights - (learning_rate * gradients) 
```

- 24) What does view do in PyTorch?

Change dims of a tensor to accomplish a reshaping whitout changing their values.


- 33) Show Python or pseudocode for the basic steps of a training loop.

    1. For each epoch
        1.1 Get i mini-batch
        1.2 Forward
        1.3 Calculate gradient
        1.4 Step (optimization, update params)
        1.5 Reset gradient values.
        1.6 Calculate accuracy on validation dataset.
