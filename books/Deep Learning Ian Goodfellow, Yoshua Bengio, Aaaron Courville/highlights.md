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