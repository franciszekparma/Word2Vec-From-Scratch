import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import EMB_DIM, DEVICE
from data_prep import build_data
from train import train_model


class Word2Vec(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, center_emb, y_embds):
    return center_emb @ y_embds


def main():
  vocab, all_sentences, all_words_in_sen = build_data()
  
  target_embds = {word: torch.randn((1, EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}
  contex_embds = {word: torch.randn((1, EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}
  
  model = Word2Vec().to(DEVICE)
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
  loss_fn = nn.CrossEntropyLoss()
  
  train_model(
    vocab, 
    all_sentences,
    all_words_in_sen,
    target_embds,
    contex_embds, 
    model,
    optimizer,
    loss_fn,
    EPOCHS
    )
  
  
if __name__ == '__main__':
  main()