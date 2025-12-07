import torch
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SHOW_DATA_STATS = False

EMB_DIM = 32
WINDOW = 2

EPOCHS=32

def sample_negatives(vocab, n, context_words, center_word):
  neg_vocab = vocab.copy()
  
  context_words = list(set(context_words))
  
  neg_vocab.pop(center_word, None)
  for word in context_words:
    neg_vocab.pop(word, None)
  
  vocab_keys = list(neg_vocab.keys())
  vocab_len = len(vocab_keys)
  
  negatives = []
  for i in range(n):
    negatives.append(vocab_keys[random.randint(0, vocab_len - 1)])
    
  return negatives