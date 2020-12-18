""" We are going to build a deep neural network to predict the sentiment
of a consumer review (positive or negative).

Dataset: Summary of people's review about the products on the Amazon web site.

Sentiment, Summary
<int>(1|0), ASCII
"""
import io
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

path = '../../datasets/product_reviews_dataset.csv'
df = pd.read_csv(filepath_or_buffer=path, encoding='ISO-8859-1')
print("Columns\n", df.columns)
print("HEAD\n", df.head(11))

print("Totales\n", df.Sentiment.value_counts())

def clean_reviews(text):
    text = re.sub("[^a-zA-Z]"," ",str(text))

    return re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)

df['Summary'] = df.Summary.apply(clean_reviews)
print("HEAD\n", df.head(11))

# Define input and output
X = df.Summary
Y = df.Sentiment
tokenizer = Tokenizer(num_words=10000, oov_token='xxxxxxx')

# Fit
tokenizer.fit_on_texts(X)
X_dict = tokenizer.word_index
print("Length of X_dict is: {}".format(len(X_dict)))
print("-"*50, "\nITEMS")
# print(X_dict.items())

# So, we have more than 32000 unique words in the text. We now apply tokenization
# on entire sequences.
X_seq = tokenizer.texts_to_sequences(X)
print("Only 10/total tokens\n")
print(X_seq[:10])

# Make vectors of equal length (100) with padding
X_padded_seq = pad_sequences(X_seq, padding='post', maxlen=100)
print("Only 3/total tokens\n")
print(X_seq[:3])
print("Shape of X_seq: {}".format(X_padded_seq.shape))

# Convert the target variable Y from Panda's series object to a NumPy array
print(type(Y))
Y = np.array(Y)
Y = Y.flatten()
print(Y.shape)
print(type(Y))

# Deep Learning model
max_length = 100
vocab_size = 10000
embedding_dims = 50
num_epochs = 10

text_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=50, input_length=100),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
text_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
text_model.summary()

# Train the model
text_model.fit(x=X_padded_seq, y=Y, epochs=num_epochs)

# Now that the model is trained, we can extract the embeddings from the model
# Each embedding is a vector of size 50, and we have 10000 embeddings.
embeddings = text_model.layers[0]
print(embeddings.weights)
rx = input("-"*42)

weights = embeddings.get_weights()[0]
print(weights.shape)

# Visualize embeddings in 3D space, we must reverse the key value for embedding
# ans respective words.

index_based_embedding = dict([(value, key) for (key, value) in X_dict.items()])

def decode_review(text):
    return ' '.join([index_based_embedding.get(i, '?') for i in text])

print('index_based_embedding[1]', index_based_embedding[1])
print('index_based_embedding[2]', index_based_embedding[2])
print('weights[1]', weights[1])

# Extract the embeddings value and put it into a .tsv file, along with another
# .tsv file that captures the words of the embedding.
vec = io.open('tsv_files/embedding_vectors_new.tsv', 'w', encoding='utf-8')
meta = io.open('tsv_files/metadata_new.tsv', 'w', encoding='utf-8')

for i in range(1, vocab_size):
    word = index_based_embedding[i]
    embedding_vec_values = weights[i]
    meta.write(word + "\n")
    vec.write('\t'.join([str(x) for x in embedding_vec_values]) + "\n")

meta.close()
vec.close()
