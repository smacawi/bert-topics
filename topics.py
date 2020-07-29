import errno
import json
import os
import pickle
import sys

from BertTM import *

def main():
    ngram = (1, 3)
    n_topics = 9

    hashtags = ['nlwhiteout', 'nlweather', 'newfoundland', 'nlblizzard2020', 'nlstorm2020',
     'snowmaggedon2020', 'stormageddon2020', 'snowpocalypse2020', 'snowmageddon',
     'nlstorm', 'nltraffic', 'nlwx', 'nlblizzard']
    stopwords = []
    if sys.argv[1] == 'hashtags':
        stopwords = get_stopwords(hashtags = hashtags)
        hashtags_ext = '_hashtags'
    elif sys.argv[1] == 'no_hashtags':
        stopwords = get_stopwords()
        hashtags_ext = '_no_hashtags'
    embedding_paths = 'bert_embedders'
    topic_paths = 'topic_models'
    for emb_file in os.listdir('bert_embedders'):
        if emb_file.endswith('.pkl'):
            
            # Load embedding and attention data
            all_model_data = pickle.load(open(f'{embedding_paths}/{emb_file}', 'rb'))
            texts, _, attentions, rows = zip(*all_model_data)
            
            # Run and save kmeans model
            labels, kmeans = get_clusters(rows, n_topics)
            emb_folder = emb_file.split('.')[0]
            directory = f'{topic_paths}/{emb_folder}{hashtags_ext}'
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    os.makedirs(f'{directory}/topics')
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            pickle.dump(kmeans, open(f'{directory}/kmeans.pkl', 'wb'))
            pickle.dump(labels, open(f'{directory}/labels.pkl', 'wb'))
             
            filtered_a, filtered_t, filtered_l = filter_data(attentions, stopwords, labels)
            features = get_phrases(filtered_t, min_count=10, threshold=0.5)
            pickle.dump(features, open(f'{directory}/features.pkl', 'wb'))
            
            components, words_label = determine_cluster_components(filtered_l, filtered_a, ngram, features)
            tfidf_indexed = tf_icf(words_label, n_topics)
            components_tfidf, components_tfidf_attn = get_tfidf_components(components, tfidf_indexed)
            
            topics_attn = topics_df(topics = n_topics, components = components, n_words = 10)
            pickle.dump(topics_attn, open(f'{directory}/topics/topics_attn.pkl', "wb"))
            topics_attn.to_csv(f'{directory}/topics/topics_attn.csv', index=False)
            
            topics_tfidf = topics_df( topics = n_topics, components = components_tfidf, n_words = 10)
            pickle.dump(topics_tfidf, open(f'{directory}/topics/topics_tfidf.pkl', "wb"))
            topics_tfidf.to_csv(f'{directory}/topics/topics_tfidf.csv', index=False)
            
            topics_tfidf_attn = topics_df(topics = n_topics, components = components_tfidf_attn, n_words = 10)
            pickle.dump(topics_tfidf_attn, open(f'{directory}/topics/topics_tfidf_attn.pkl', "wb"))
            topics_tfidf_attn.to_csv(f'{directory}/topics/topics_tfidf_attn.csv', index=False)

def get_stopwords(hashtags = [], filename = 'stopwords-en.json'):
    with open(filename) as fopen:
        stopwords = json.load(fopen)

    stopwords.extend(['#', '@', '…', "'", "’", "[UNK]", "\"", ";", "*", "_", "amp", "&", "“", "”"] + hashtags)
    return(stopwords)

if __name__ == '__main__':
    main()
