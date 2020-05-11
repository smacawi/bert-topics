from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import os
import pandas as pd
import pickle

def main():
    base_path = "topic_models/sent_embedder/"
    output_coherence(f"{base_path}features/features_sent_embeddings.pickle",
                     f"{base_path}topics",
                     embed_name = "sent_embedder",
                     coherence_type = ['c_v', 'c_uci', 'c_npmi', 'u_mass'])

def get_coherence(features, topics, coherence_type = 'c_v'):
    dct = Dictionary(features)
    topics = topics.values.tolist()

    if coherence_type == 'u_mass':
        bow_corpus = [dct.doc2bow(f) for f in features]
        cm = CoherenceModel(topics=topics, corpus=bow_corpus, dictionary=dct, coherence=coherence_type)
        coherence = cm.get_coherence()  # get coherence value

    elif coherence_type in ['c_v', 'c_uci', 'c_npmi']:
        cm = CoherenceModel(topics=topics, texts = features, dictionary=dct, coherence=coherence_type)
        coherence = cm.get_coherence()  # get coherence value

    else:
        print(f"'{coherence_type}' is not a coherence model. Use one of the following arguments: 'u_mass', 'c_v', 'c_uci', 'c_npmi'.")
    
    return coherence

def output_coherence(features_path, topics_dir, embed_name, coherence_type, topic_model = "kmeans"):
    topic_paths = []
    features = pickle.load(open(features_path, 'rb'))
    df = {'embeddings': [], 'model': [], 'components':[], 'topics': [], 
          'ngrams_per_topic':[],'ct': [], 'coherence': [], 'hashtags': []}

    for file in os.listdir(topics_dir):
        topic_paths.append((f"{topics_dir}/{file}", file))

    for path in topic_paths:
        topics = pickle.load(open(path[0], 'rb'))
        for ct in coherence_type:
            coherence = get_coherence(features, topics, coherence_type = ct)
            df['embeddings'].append(embed_name)
            df['model'].append(topic_model)
            df['components'].append(path[1].split('.')[0])
            df['topics'].append(10)
            df['ngrams_per_topic'].append(10)
            df['ct'].append(ct)
            df['coherence'].append(coherence)
            df['hashtags'].append('no')
    df = pd.DataFrame.from_dict(df).groupby('ct', group_keys = False).apply(
        pd.DataFrame.sort_values, 'coherence', ascending=False).reset_index(drop = 'True')
    df.to_csv("coherence_scores.csv", index=False)

if __name__ == '__main__':
    main()

