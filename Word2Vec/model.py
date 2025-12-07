import torch
import torch.nn.functional as F

from utils import sample_negatives, EMB_DIM, WINDOW, DEVICE
from data_prep import build_data

def main():
  vocab, all_sentences, all_words_in_sen = build_data()
  
  target_embds = {word: torch.randn((1, EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}
  contex_embds = {word: torch.randn((1, EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}
  
  for s, sentence in enumerate(all_sentences):
    for w, word in enumerate(all_words_in_sen):
      w_l, w_r = len(all_words_in_sen[:w]), len(all_words_in_sen[w:]) #num of the words to the left and to the right from the current worod
      
      y_pos_words = all_words_in_sen[max(0, w - WINDOW) : w] + all_words_in_sen[w + 1 : w + WINDOW + 1] #the words that we want the target to be simillar to
      y_pos_embds = [contex_embds[x] for x in y_pos_words]
      
      y_neg_words = sample_negatives(vocab, n=2*WINDOW, context_words=y_pos_words, center_word=word)
      y_neg_embds = [contex_embds(x) for x in y_neg_words]
      
      center_word = word
      center_emb = target_embds[word] #the word embedding that we are currently at
      
      
      
      
  
  
if __name__ == '__main__':
  main()