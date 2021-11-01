import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# https://pytorch.org/text/stable/vocab.html
# https://pytorch.org/text/stable/data_utils.html
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

"""
AG News dataset contains text from 127600 online news articles, from 4 different
categories: World, Sports, Business, and Science/Technology. AG News is typically
used for topic classification: given an unseen news article, we're interested
in predicting the topic.
"""

dataset_url = '../../../../../../ML_DL/datasets/agnews'

train_iter, test_iter = torchtext.datasets.AG_NEWS(
    root=dataset_url,
    split=('train', 'test')
)

# Build the vocabulary with the raw training dataset
tokenizer = get_tokenizer(tokenizer='basic_english', language='en')

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# The vocabulary block converts a list of tokens into integers.
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
# unk = unknown (default)
vocab.set_default_index(vocab['<unk>'])

# example_words = ['cat', 'dog', 'chicken']
# print(example_words, vocab(example_words))

# Create data loaders
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_batch(batch):
  labels = torch.tensor([label_pipeline(example[0]) for example in batch])
  sentences = [torch.tensor(text_pipeline(example[1])) for example in batch]
  data = pad_sequence(sentences).clone().detach()

  return [data, labels]

train_iter, test_iter = torchtext.datasets.AG_NEWS(
    root=dataset_url,
    split=('train', 'test')
)

train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

BATCH_SIZE = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

train_iter, test_iter = torchtext.datasets.AG_NEWS(
    root=dataset_url,
    split=('train', 'test')
)

num_classes = len(set([label for (label, text) in train_iter]))
