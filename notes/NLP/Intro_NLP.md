Text data can either be in structured or unstructured form.

## Tokenization
Tokens stand for individual words/symbols/numbers present in the text.

-----------------------------

There are some simpler and much more complicated methods, in addition to representing text in numerical form. We can basically divide them into 2 categories.

1. Frequency-based
2. Prediction-based

Frequency-based techniques include vectorizer, tf-idf, and hashing vectorizer, whereas prediction-based techniques involve such methods as CBOW (continuous bag-of-words) and Skip-Gram model.

## Embeddings
Embeddings are, again, numerical representations of text information, but they are much more powerful and relevant, compared to other methods. Embeddings are nothing but the weights of the hidden layer of a shallow neural network that was trained on a certain set of text.

The core advantage that word embedding offers is that it captures the semantic meaning of the word, as it uses the idea of distributed representations. It predicts these embedding values, based on other words similar to that word.
