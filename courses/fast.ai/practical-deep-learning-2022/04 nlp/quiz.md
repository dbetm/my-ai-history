- 1) What is self-supervised learning?

When the independent variable is implicit in the dependent variable. This mean, the algorithm can learn patterns (relationships) of the inputs. Example, a language model that predicts the next words.


- 4) What are self-supervised models usually used for?

To generate useful representations of the input data. So the model can be used as a pretraining model to perform transfer learning and accomplish other goals.


- 5) Why do we fine-tune language models?

To take advantages of what a pre-trained model has learned, for example, the English syntax.


- 6) What are the three steps to create a state-of-the-art text classifier?
    - Train a model in a huge amount of text.
    - Fine-tuning the pre-trained model using the domain specific dataset.
    - Create classifier.


- 8) What are the three steps to prepare your data for a language model?
    - Tokenization
    - Map tokens to integers
    - Create batches


- 9) What is "tokenization"? Why do we need it?

It's the process of break the text into units, it can be subwords, letters or the whole words.
We need it to be able to train a neural network feeding numbers, so each unit represents a number.


- 10) Name three different approaches to tokenization.

    - Word tokenization
    - Subword tokenization
    - Character-based tokenization


- 17) Why do we need padding for text classification? Why don't we need it for language modeling?

To have inputs of the same lenght in a batch, it's the way PyTorch NN expect the input.
In the other case, because in language modeling we can separate a token stream into batches, where it's important to keep order, since we are treating the input as a sequence.


- 18) What does an embedding matrix for NLP contain? What is its shape?

For each level (row) there's an item of the vocab, it's the representation of the token 
in a numerical vector. The width it's a fixed lenght, and the goal is to tweak (through training) 
the numbers to appropiately capture the semantic relations between words.

During training, the embedding matrix is updated along with the rest of the model parameters to minimize the loss function specific to the given task. However, pre-trained word embeddings are often used as a starting point, leveraging large-scale datasets like Wikipedia or Twitter, which capture general word meanings and semantic relationships.


- 19) What is "perplexity"?

It's a metric used on NLP to measure how "surprised" a model is when it encounters a new unseen sequence of words. A lower perplexity is better.


- 22) Why is text generation always likely to be ahead of automatic identification of machine-generated texts?

Because better automatic identification machine-generated texts (discriminator) can be used to generate better text generation models.
