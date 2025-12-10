import torch
from torch import nn

from utils import EMB_DIM, EPOCHS, LR, WEIGHT_DECAY, LOAD_CHECKPOINT, PATH_CHECHPOINT, DEVICE, SEED
from data_prep import build_data
from train import train_model

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class Word2Vec(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, center_emb, y_embds):
    return center_emb @ y_embds.T


vocab, all_words_in_sen = build_data()

embds = {word: torch.randn((EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}


def main():
  global embds
  
  if LOAD_CHECKPOINT:
    print("Loading checkpoint...")
    checkpoint = torch.load(PATH_CHECHPOINT, map_location=DEVICE, weights_only=False)
    embds = checkpoint['embeddings']
    optimizer = torch.optim.AdamW(list(embds.values()), lr=LR, weight_decay=WEIGHT_DECAY) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  else:
    optimizer = torch.optim.AdamW(list(embds.values()), lr=LR, weight_decay=WEIGHT_DECAY)
     
  model = Word2Vec().to(DEVICE)
  
  loss_fn = nn.BCEWithLogitsLoss()
  
  train_model(
    vocab,
    all_words_in_sen,
    embds,
    model,
    optimizer,
    loss_fn,
    EPOCHS
    )
 
if __name__ == '__main__':
  main()
