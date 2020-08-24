import errno
import json
import os
import pickle
import sys

from BertTM import *

def main():
    ngram = (1, 3)
    n_topics = [20]
    model_type = "nlwx"

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
    embedding_paths = f'bert_embedders/{model_type}'
    topic_paths = f'topic_models/{model_type}'
    for emb_file in os.listdir(embedding_paths):
        if emb_file.endswith('clstr_emb_att-nlwx_ht_no_rt-l2_all_events_balanced.pkl'):
            
            # Load embedding and attention data
            all_model_data = pickle.load(open(f'{embedding_paths}/{emb_file}', 'rb'))
            texts, preds, attentions, rows = zip(*all_model_data)
            if type(preds[0]) is dict:
                preds = [max(d, key=d.get) for d in preds]
            
            for t in n_topics:
                # Run and save kmeans model
                print(f"Running kmeans model for {emb_file} with {t} topics.")
                labels, kmeans = get_clusters(rows, t)
                emb_folder = emb_file.split('.')[0]
                directory = f'{topic_paths}/{emb_folder}{hashtags_ext}'
                if not os.path.exists(f'{directory}/{t}'):
                    try:
                        #os.makedirs(directory)
                        os.makedirs(f'{directory}/{t}')
                        os.makedirs(f'{directory}/{t}/topics')
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                print(f"Saving kmeans model and labels for {emb_file} with {t} topics.")
                pickle.dump(kmeans, open(f'{directory}/{t}/kmeans.pkl', 'wb'))
                pickle.dump(labels, open(f'{directory}/{t}/labels.pkl', 'wb'))
                pickle.dump(preds, open(f'{directory}/{t}/preds.pkl', 'wb'))
                
                print(f"Getting featuers for kmeans model for {emb_file} with {t} topics.")
                filtered_a, filtered_t, filtered_l = filter_data(attentions, stopwords, labels)
                features = get_phrases(filtered_t, min_count=10, threshold=0.5)
                pickle.dump(features, open(f'{directory}/{t}/features.pkl', 'wb'))
                
                print(f"Getting components for kmeans model for {emb_file} with {t} topics.")
                components, words_label = determine_cluster_components(filtered_l, filtered_a, ngram, features)
                # Hackey solution when a label has no words
                for n in range(t):
                    if n not in words_label.keys():
                        words_label[n] = [""]
                        components[n] = {}
                        components[n][""] = 0
                tfidf_indexed = tf_icf(words_label, t)
                components_tfidf, components_tfidf_attn = get_tfidf_components(components, tfidf_indexed)

                print(f"Determining topics for kmeans model for {emb_file} with {t} topics.")
                save_topics(emb_file, t, directory, components, "topics_attn")
                save_topics(emb_file, t, directory, components, "topics_tfidf")
                save_topics(emb_file, t, directory, components, "topics_tfidf_attn")

def save_topics(emb_file, t, directory, components, filename, n_words = 10):
    print(f"Determining topics for kmeans model for {emb_file} with {t} topics.")
    topics = topics_df(topics = t, components = components, n_words = n_words)
    pickle.dump(topics, open(f'{directory}/{t}/topics/{filename}.pkl', "wb"))
    topics.to_csv(f'{directory}/{t}/topics/{filename}.csv', index=False)

                
def get_stopwords(hashtags = [], filename = 'stopwords-en.json'):
    with open(filename) as fopen:
        stopwords = json.load(fopen)

    stopwords.extend(['#', '@', '…', "'", "’", "[UNK]", "\"", ";", "*", 
                      "_", "amp", "&", "“", "”", "rt"] + hashtags)
    return(stopwords)

if __name__ == '__main__':
    main()
