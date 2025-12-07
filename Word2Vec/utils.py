import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SHOW_DATA_STATS = False
EMB_DIM = 32
WINDOW = 2


def get_random_word(vocab, n, context_words, center_word):
  