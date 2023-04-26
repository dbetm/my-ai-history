## Batch gradient descent
It's about updating the weights after the whole observations are forwad propagated in the neural network

## Stochastic Gradient Descent (SGD)
It's about updating the weights after each observation is forwad propagated.

## Mini Batch Stochastic Gradient Descent
It's prefer on practice, combine the previouos two ones.

## Backpropation
It's about updating the weights in an efficient way using the chain rule, after the loss function is evaluated.

## Training the ANN with SGD

1) Randomly initialize the weights to small numbers close to 0 (but not 0).
2) Input the first observation of your dataset in the input layer, each feature in one input node-
3) Forward-propagation: from left-to-right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted result y.
4) Compare the predicted result to the expected results. Measure the error.
5) Backpropagation: from right-to-left propagate the error to update the weights, using the calculeted gradients and a hyperparameter to control the learning rate.
6) Repeat steps 1-5 for each observation.
7) That's one epoch, redo for enough epochs.