## Bag-of-words model

The idea is to have a vector representing the frequencies of words of a given text, this vector is initialized with zeros at the beginning, with a lenght matching the size of the set of words that we are considering, example: 20k for most popular vocabulary in English.

**Example**
If we have training data representing the text for too many emails, and we label each text one with yes or not, we can use a logistic regresion model to classify any email text.