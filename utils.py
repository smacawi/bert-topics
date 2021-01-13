import json
import pandas as pd

# Import stopwords and hashtags to drop when deternmining topic cluster components.
# These stopwords will _not_ be dropped from the actual BERT models when determining topic clusters.
def get_stopwords(hashtags = [], filename = 'data/stopwords-en.json'):
    with open(filename) as fopen:
        stopwords = json.load(fopen)

    stopwords.extend(['#', '@', '…', "'", "’", "[unk]", "\"", ";", 
                      "*", "_", "amp", "&", "“", "”"] + hashtags)
    return(stopwords)

# Create dataframe from topics  
# Modified from code in block 28 of commit 9895ee0 at:
# https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
def topics_df(topics, components, n_words = 20):
    df = {}
    for i in range(topics):
        words = sorted(components[i], key=components[i].get, reverse=True)[:n_words]
        df['topic %d' % (i)] = words
        if len(words) < n_words:
            df['topic %d' % (i)].extend([''] * (n_words - len(words)))
    return pd.DataFrame.from_dict(df)

