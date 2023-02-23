2) Name five areas where deep learning is now the best in the world.
    - Image recognition on ImageNet
    - Cancer image recognition
    - Translating between many languages
    - Natural Language processing (answering questions, summarazing, etc.)
    - Playing games such as go and chess.

4) What are the requirements for parallel distributed processing (PDP)?
    - Connectivity between neural units.
    - A set of processing units.
    - A state of activation.
    - A propagation rule for propagation patterns of activities through the network.
    - A learnig rule (patterns of connectivity are modified by experience).
    - An environment within the system must operate.

5) What were the two theoretical misunderstandings that held back the field of neural networks?
    - Insuficcient amount of computing to perform on neural networks with more than 2 layers.
    - The fact that complex problem could be solved using more layers on neural networks.

10) Why is it hard to use a traditional computer program to recognize images in a photo?
    - There are too many rules to consider. And they are different from image to image. And at the end we don't know very good how we humans beings recognize a particular photo (thus, it's difficult to figure out the bunch of mencioned rules).


14) Why is it hard to understand why a deep learning model makes a particular prediction?
    - Just because there are too many model parameters involved. But, for example for CNNs it could be easy to map activations an then understand what features are recognized layer by layer, helping to understand the final prediction.

15) What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
    - Universal Approximation Theorem.


21) What will fastai do if you don't provide a validation set?
    - It will create one by default. Using 20% of the whole data.


22) Can we always use a random sample for a validation set? Why or why not?
    - No, sometimes it's hard, because we have too little data. Use too little data it could be wrong because the different data distribution.

24) What is a metric? How does it differ from "loss"?
    - A metric help us to measure the performance of the model on an specific way. The loss could be a metric, but the distinction is that the loss is used the training process to update the weights of the model (using SGD behind scenes).

29) What is an "architecture"?
    - The organization of layers, unit per layers, feedback and so on.


33) What's the best way to avoid failures when using AI in an organization?
    - Using a test set besided training and validation data sets.
