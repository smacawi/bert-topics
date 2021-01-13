'''Use the Gensim module CoherenceModel() to calculate coherence for different models.
The coherence types used by default are c_v and c_npmi.
Outputs a csv file for each coherence type and for each topic number (5, 9, 10, 15).

See Gensim documentation for more details: https://radimrehurek.com/gensim/models/coherencemodel.html
'''

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import os
import pandas as pd
import pickle
import re

def main():
    '''
    Iterate through different topic models, coherence types and number of topics.
    Agregate by topic model and output csv for each combination of coherence types and number of topics.
    '''
    
    base_path = f"topic_models/nlwx/"
    topics = [5,9,10,15]
    _, dirs, _ = next(os.walk(base_path))
    dirs = list(d for d in dirs if (not d[0] == '.') and (not d[0].startswith('t')))
    ct = ['c_v','c_npmi']
    for t in topics:
        for c in ct:
            c_dfs = []
            for d in dirs:
                try:
                    df = output_coherence(f"{base_path}{d}/{t}/features.pkl",
                                          f"{base_path}{d}/{t}/topics",
                                          embed_name = d,
                                          coh_type = c,
                                          n_topics = t)
                    c_dfs.append(df)
                    print(len(c_dfs))
                except:
                    print(f"Model {d} not found.")
            try:
                output = pd.concat(c_dfs).groupby('ct', group_keys = False).apply(
                pd.DataFrame.sort_values, 'coherence', ascending=False).reset_index(drop = 'True')
                #output.to_csv(f"outputs/coherence/coherence_scores_nlwx_{t}-topics_{c}-coherence.csv", index=False)
                print(f"Succesfully saved {t} topics and {c} coherence.")
            except:
                print(f"Failed to save {t} topics and {c} coherence.")
                continue

def return_coherence(features, topics, coherence_type):
    '''Returns coherence score for topics using the Gensim module CoherenceModel().
    For details, see https://radimrehurek.com/gensim/models/coherencemodel.html

        Parameters
        ----------
        features : pandas.DataFrame
            Only necessary for coherence_type = 'u_mass'
        topics : pandas.DataFrame
            Index:
                RangeIndex
            Columns:
                Name: Date, dtype: datetime64[ns]
                Name: Integer, dtype: int64
        coherence_type : str
            'u_mass', 'c_v', 'c_uci', 'c_npmi'

        Returns
        -------
        coherence : float
            Value of coherence
        '''
    
    dct = Dictionary(features)
    topics = topics.T.values.tolist()

    if coherence_type == 'u_mass':
        try:
            bow_corpus = [dct.doc2bow(f) for f in features]
            cm = CoherenceModel(topics=topics, corpus=bow_corpus, dictionary=dct, coherence=coherence_type)
            coherence = cm.get_coherence()  # get coherence value
        except:
            coherence = 0

    elif coherence_type in ['c_v', 'c_uci', 'c_npmi']:
        try:
            cm = CoherenceModel(topics=topics, texts=features, dictionary=dct, coherence=coherence_type)
            coherence = cm.get_coherence()  # get coherence value
        except:
            coherence = 0

    else:
        print(f"'{coherence_type}' is not a coherence model. Use one of the following arguments: 'u_mass', 'c_v', 'c_uci', 'c_npmi'.")
    
    return coherence

# Gather several coherence scores from different metrics into one dataframe.
def output_coherence(features_path, topics_dir, embed_name, coh_type, n_topics, topic_model = "kmeans"):
    '''Returns coherence score for topics using the Gensim module CoherenceModel().
    For details, see https://radimrehurek.com/gensim/models/coherencemodel.html

        Parameters
        ----------
        features_path : str
            Only necessary for coherence_type = 'u_mass'
        topics_dir : str
        embed_name : str
        coh_type : str
        n_topics : int
        topic_model : str
            default : "kmeans"

        Returns
        -------
        coherence : pandas.DataFrame
            Index:
                RangeIndex
            Columns:
                Name: embeddings, dtype: str
                Name: model, dtype: int64
                Name: components, dtype: str
                Name: topics, dtype: int64
                Name: ngrams_per_topic, dtype: datetime64[ns]
                Name: ct, dtype: int64
                Name: coherence, dtype: datetime64[ns]
                Name: hashtags, dtype: str
                Name: phrasing, dtype: str
                Name: max_df, dtype: str
                Name: stf, dtype: bool
                Name: ngrams, dtype: str
        '''
    topic_paths = []
    features = pickle.load(open(features_path, 'rb'))
    df = {'embeddings': [], 'model': [], 'components':[], 'topics': [], 
          'ngrams_per_topic':[],'ct': [], 'coherence': [], 'hashtags': [],
          'phrasing':[],'max_df':[],'stf':[],'ngrams':[]}
    ht = test_ht(embed_name)
    p = test_phrasing(embed_name)
    m = get_mtf(embed_name)
    s = get_stf(embed_name)
    n = get_ngram(embed_name)
    
    embedding = embed_name.split('_ngram')[0]
    for file in os.listdir(topics_dir):
        if file.endswith(".pkl"):
            topic_paths.append((f"{topics_dir}/{file}", file))

    for path in topic_paths:
        topics = pickle.load(open(path[0], 'rb'))
        coherence = return_coherence(features, topics, coherence_type = coh_type)
        comps = path[1].split('.')[0]
        print(f"Getting {coh_type} coherence for {embed_name} with {n_topics} topics and {comps} components.")
        df['embeddings'].append(embedding)
        df['model'].append(topic_model)
        df['components'].append(comps)
        df['topics'].append(n_topics)
        df['ngrams_per_topic'].append(10)
        df['ct'].append(coh_type)
        df['coherence'].append(coherence)
        df['hashtags'].append(ht)
        df['phrasing'].append(p)
        df['max_df'].append(m)
        df['stf'].append(s)
        df['ngrams'].append(n)
    df = pd.DataFrame.from_dict(df)
    return df

def test_ht(embed_name):
    '''Check from embedding path whether topics include hashtags.
    Parameters
    ----------
        embed_name : str
    Returns
    -------
        ht : str
    '''
    if 'no_hashtag' in embed_name:
        ht = 'no'
    else:
        ht = 'yes'
    return ht

def test_phrasing(embed_name):
    '''Check from embedding path whether topics include phrasing.
    Parameters
    ----------
        embed_name : str
        
    Returns
    -------
        p : str
    '''
    if 'no_phrasing' in embed_name:
        p = 'no'
    else:
        p = 'yes'
    return p

def get_mtf(embed_name):
    '''Check from embedding path what mdf value was used for tf-idf.
        Parameters
        ----------
        embed_name : str
        
        Returns
        -------
        mdf : str
    '''
    
    if 'mdf' in embed_name:
        r = re.findall("\d+\.\d+", embed_name)[0]
    else:
        r = str(1.0)
    return r

def get_stf(embed_name):
    '''Check from embedding path whether stf value was used for tf-idf.
        Parameters
        ----------
        embed_name : str
        
        Returns
        -------
        s : str
    '''
    
    if 'stf' in embed_name:
        if "True" in embed_name:
            s = True
        else:
            s = False
    else:
        s = False
    return s

def get_ngram(embed_name):
    '''Check from embedding path whether stf value was used for tf-idf.
        Parameters
        ----------
        embed_name : str
        
        Returns
        -------
        s : str
    '''
    
    n = embed_name.split("ngram",1)[1][0]
    return n

if __name__ == '__main__':
    main()

