import torch
import torch.nn.functional as F

from utils import EMB_DIM, WINDOW, DEVICE
from data_prep import build_data

def main():
  vocab, all_sentences = build_data()
  
  target_embds = {word: torch.randn((1, EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}
  contex_embds = {word: torch.randn((1, EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}
  
  
  
  
  
  
  
if __name__ == '__main__':
  main()