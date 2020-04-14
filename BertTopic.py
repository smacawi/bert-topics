import json
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn

from BertForSequenceClassificationOutputPooled import *
from BertTM import *
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sklearn.cluster import KMeans

with open('stopwords-en.json') as fopen:
    stopwords = json.load(fopen)

batch_size = 20
ngram = (1, 3)
n_topics = 10

all_model_data = pickle.load(open("attentions_sent_embeddings.pkl", "rb"))
_, _, attentions, rows = zip(*all_model_data)

print("Fitting kmeans model.")
start_time = time.time()
kmeans = KMeans(n_clusters = n_topics, random_state = 0).fit(rows)
labels = kmeans.labels_
print("--- %s seconds ---" % (time.time() - start_time))

overall, filtered_a = [], []
print("Filtering attentions.")
for a in attentions:
    f = [i for i in a if i[0] not in stopwords]
    overall.extend(f)
    filtered_a.append(f)

print("Generating ngrams.")
o_ngram = generate_ngram(overall, ngram)
features = []
for i in o_ngram:
    features.append(' '.join([w[0] for w in i]))
features = list(set(features))

print(
"""
Determining cluster components. This will take some hours. 
Progress will be printed for every 100th processed document.
""")

start_time = time.time()
components = np.zeros((n_topics, len(features)))
for no, i in enumerate(labels):
    if (no + 1) % 100 == 0:
        print('Processed %d'%(no + 1))
    f = generate_ngram(filtered_a[no], ngram)
    for w in f:
        word = ' '.join([r[0] for r in w])
        score = np.mean([r[1] for r in w])
        if word in features:
            components[i, features.index(word)] += score
            
print("Finished determining cluster components.")
print("--- %s seconds ---" % (time.time() - start_time))
pickle.dump(components, open("components_sent_embed.pickle", "wb"))

topics <- print_topics_modelling(
    10,
    feature_names = np.array(features),
    sorting = np.argsort(components)[:, ::-1],
    n_words = 10,
    return_df = True,
)

pickle.dump(topics, open("topics_sent_embed.pickle", "wb"))