from models.BertEmbedder import BertEmbedder

import pandas as pd
import sys

model_dir = sys.argv[1]
outfile = sys.argv[2]

df = pd.read_csv("data/nlwx_2020_hashtags_no_rt_predictions.csv")
texts = df['text']
labels = df['prediction']

bert_embedder = BertEmbedder(model_dir, sentence_bert=True)
bert_embedder.get_embeddings(texts)
bert_embedder.save_outputs(texts, labels, outfile)