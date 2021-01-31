import errno
import json
import os
import pickle
import sys

from models.BertTopicModel import BertTopicModel
from utils import get_stopwords, topics_df

class TopicSearch():
    def __init__(self,
                 embedding_paths = '../data/embeddings/',
                 topic_paths = '../data/topicsearch'):
        self.topic_paths=topic_paths
        self.embedding_paths = embedding_paths
        self.hyperparams = {}
        #self.z = zipfile.ZipFile('data/topicsearch.zip', 'w', zipfile.ZIP_DEFLATED)

    def search(self,
               ngrams=[1,2], 
               max_df=[0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
               stf=[True, False],
               n_topics = [5,10,15],
               hashtags = [],
               hashtag_opts = [True, False],
               phrasing = [True, False],
               p_min_count=5,
               p_threshold=100):

        for emb_file in os.listdir(self.embedding_paths):
            if emb_file.endswith('.pkl'):
                for t in n_topics:
                    self._init_model(emb_file, t)
                    for h in hashtag_opts:
                        stopwords = get_stopwords(hashtags = hashtags,filename = '../data/stopwords-en.json')
                        for n in ngrams:
                            ngram = (1, n)
                            for s in stf:
                                for m in max_df:
                                    for p in phrasing:
                                        self._create_dirs(t,h,n,p,s,m,emb_file)
                                        
                                        print(f'Saving kmeans model labels for {emb_file} with '
                                              f'{t} topics, phrasing {p}, {n} ngrams, mdf {m} and stf {s}.')
                                        pickle.dump(self.labels, open(f'{self.directory}/{t}/labels.pkl', 'wb'))
                                        
                                        print(f'Getting features for kmeans model for {emb_file} with '
                                              f'{t} topics, phrasing {p}, {n} ngrams, mdf {m} and stf {s}.')
                                        features = self.bertTM.get_features(stopwords=stopwords,
                                                                            phrasing=p,
                                                                            min_count=p_min_count, 
                                                                            threshold=p_threshold)
                                        pickle.dump(features, open(f'{self.directory}/{t}/features.pkl', 'wb'))
                                        
                                        print(f'Getting components for kmeans model for {emb_file} with '
                                              f'{t} topics, phrasing {p}, {n} ngrams, mdf {m} and stf {s}.')
                                        components, words_label = self.bertTM.determine_cluster_components(ngram)
                                        components_tfidf, components_tfidf_attn = self.bertTM.get_tfidf_components(max_df = m,
                                                                                                                   stf = s)
                                        
                                        print(f'Determining topics for kmeans model for {emb_file} with '
                                              f'{t} topics, phrasing {p}, {n} ngrams, mdf {m} and stf {s}.')
                                        self._save_topics(emb_file, t, components, "topics_attn")
                                        self._save_topics(emb_file, t, components_tfidf, "topics_tfidf")
                                        self._save_topics(emb_file, t, components_tfidf_attn, "topics_tfidf_attn")
                                        self._save_hyperparams(emb_file,t,h,s,m,p)

    def _save_hyperparams(self,emb_file,t,h,s,m,p):
        self.hyperparams['emb_file'] = emb_file.split('.')[0]
        self.hyperparams['topics'] = t
        self.hyperparams['hashtags'] = h
        self.hyperparams['stf'] = s
        self.hyperparams['max_df'] = m
        self.hyperparams['phrasing'] = p
        with open(f'{self.directory}/{t}/hyperparams.json', 'w') as f:
            json.dump(self.hyperparams, f)
    
    def _init_model(self, emb_file, t):
        # Load embedding and attention data
        all_model_data = pickle.load(open(f'{self.embedding_paths}/{emb_file}', 'rb'))
        texts, _, attentions, embeddings = zip(*all_model_data)
        # Run kmeans model
        print(f"Running kmeans model for {emb_file} with {t} topics.")
        self.bertTM = BertTopicModel(
            texts = texts, 
            attentions = attentions, 
            embeddings = embeddings,
            n_clusters = t)
        self.labels = self.bertTM.get_clusters()
                 
    def _create_dirs(self,t,h,n,p,s,m,emb_file):
        self.emb_folder = f"{emb_file.split('.')[0]}_ngram{n}_phrasing-{p}_stf-{s}_mdf-{m}"
        self.directory = f'{self.topic_paths}/{self.emb_folder}_hashtags-{h}'
        if not os.path.exists(f'{self.directory}/{t}'):
            try:
                os.makedirs(f'{self.directory}/{t}')
                os.makedirs(f'{self.directory}/{t}/topics')
            except OSError as exc: # Guard against raise condition
                if exc.errno != errno.EEXIST:
                    raise
    
    def _save_topics(self, emb_file, t, components, filename, n_words = 10):
        print(f"Determining topics for kmeans model for {emb_file} with {t} topics.")
        topics = topics_df(topics = t, components = components, n_words = n_words)
        pickle.dump(topics, open(f'{self.directory}/{t}/topics/{filename}.pkl', "wb"))
        topics.to_csv(f'{self.directory}/{t}/topics/{filename}.csv', index=False)
