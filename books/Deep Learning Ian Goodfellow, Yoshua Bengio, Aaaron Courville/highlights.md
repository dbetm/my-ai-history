# Introduction

## Representation matters
In computer science, operations such as searching a collection of data can proceed
exponentially faster if the collection is structured and indexed intelligently. 
People can easily perform arithmetic on Arabic numerals, but find arithmetic on 
Roman numerals much more time-consuming. It is not surprising that the choice 
of representation has an enormous effect on the performance of machine learning algorithms.

pp.3

-------------------------------------

## Representation learning
Use machine learning to discover not only
the mapping from representation to output but also the representation itself.

The quintessential example of a representation learning algorithm is the au
toencoder. An autoencoder is the combination of an encoder function that
converts the input data into a different representation, and a decoder function
that converts the new representation back into the original format.

pp.4

-------------------------------------

# Machine Learning Basics

## What is learning?
Mitchell (1997) provides the definition “A computer program is said to learn from 
experience E with respect to some class of tasks T and performance measure P, 
if its performance at tasks in T, as measured by P, improves with experience E”.

pp. 98

-------------------------------------

## Classification with missing inputs
When some of the inputs may be missing, rather than providing a single classification function, the learning algorithm must learn a set of functions.

pp. 100


## Capacity, overfitting, and underfitting
Machine learning algorithms will generally perform best when their capacity is appropriate for the true complexity of the task they need to perform and the amount of training data they are provided with.

pp. 112

The most important results in statistical learning theory show that the discrepancy between training error and generalization error is bounded from above by a quantity that grows as the model capacity grows but shrinks as the number of training examples increases.

pp. 114

## Hyperparameters and validation sets

Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.

pp. 120

## Unsupervised Learning Algorithms

Informally, unsupervised learning refers to most attempts to extract information from a distribution that don't require human labor to annotate examples.

pp. 146

## Stochastic Gradient Descent - SGD

The insight of SGD is that the gradient is an expectation. The expectation may be approximately estimated using a small set of samples.

pp. 153

## Challenges motivating deep learning

The core idea in DL is that we assume that the data was generated by the composition of factors or features, potentially at multiple levels in a hierarchy.

pp. 160

Most of the long sequences of letters do not correspond to a natural language sequence: the distribution of natural language sequences occupies a very small volume in the total space of sequences of letters.

pp. 162

When the data lies on a low-dimensional manifold, it can be most natural for machine learning algorithms to represent the data in terms of coordinates on the manifold, rather than in terms of coordinates in `R^n`.

---------------------------------

# Deep Feedforward Networks (also aka as MLPs)

A feedforward network defines a mapping `y = f(x, theta)` and learns the value of the parameters `theta` that result in the best function approximation.

pp. 168


## Example: Learning XOR

Much as Turing machine's memory needs only to be able to store 0 or 1 states, we can build a universal function approximator from rectified linear functions.

pp. 175

## Gradient Based Learning

Softmax functions are most often used as the output of a classifier, to represent the probability distribution over `n` different classes.

pp. 184

In general, we can think of the neural network as representing a function `f(x;θ)`.
The outputs of this function are not direct predictions of the value `y`. Instead,
`f(x;θ) = ω` provides the parameters for a distribution over `y`. Our loss function
can then be interpreted as `− log p(y; ω(x))`.

pp. 188

## Architecture Design

The Universal Approximation Theorem states that a feedforward network with a linear output layer and at least one hidden layer with any "squashing" activation function can approximate any Borel measurable function (any continous function on a closed and bounded subset of R²) from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units.

pp. 198

Empirically, greather depth does seem to result in better generalization for a wide variety of tasks.

## Back-propagation and other differentiation algorithms

In learning algos., the gradient we most often require is the gradient of the cost function with respect to the parameters.

## Historical notes

Feedforward networks can be seen as efficient nonlinear functions approximators based on using gradient descent to minimize the error in a function approximation.

pp. 224

# Regularization for Deep Learning

## Dataset Augmentation

The best way to make a machine learning model generalize better is to train it on more data. One way to get around this problem is to create fake data and add it to the training set.

pp. 240

A classifier needs to take a complicated, high dimensional input **x** and summarize it with a single category identity **y**. This means that the main task facing a classifier is to be invariant to a wide variety of transformation.

pp. 240

## Semi-supervised learning

In DL, semi-supervised learning usually refers to learning a representation `h=f(x)`.

pp. 243