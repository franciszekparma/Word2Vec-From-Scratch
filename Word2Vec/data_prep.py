from datasets import Dataset, load_dataset
import pyarrow as pa
import os

from utils import SHOW_DATA_STATS

def build_data(path="data/flickr30k_captions", show_stats=SHOW_DATA_STATS): 
  if not os.path.exists(path):
    dataset = load_dataset("embedding-data/flickr30k_captions_quintets")
    dataset.save_to_disk("data")
  else:
    print(f"Loading data...")
    
  dataset = Dataset.from_file("data/train/data-00000-of-00001.arrow")

  all_sentences = []
  for example in dataset:
    all_sentences.extend(example['set'])
  
  all_words_in_sen = []
  all_words = []
  for sentence in all_sentences:
    words_in_sen = sentence.split()
    all_words_in_sen.append(words_in_sen)
    all_words.extend(words_in_sen)
    
  vocab = list(set(all_words)) #getting only the unique words (no duplicates)

  if show_stats:
    print(f"Total sentences: {len(all_sentences)}")
    print(f"Total words (with duplicates): {len(all_words)}")
    print(f"Vocab size: {len(vocab)}")
  return vocab, all_sentences

if __name__ =='__main__':
  build_data()