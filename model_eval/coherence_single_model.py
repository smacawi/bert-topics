from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary

import os
import pandas as pd
import pickle

def main():
    """
    A function to output all the coherence scores in the Gensim library

    Attributes
    ----------
    topics : list
        a list of lists where each sublist is a list of topics
    features : list
        a list of lists where each list is the features in the input documents

    Output
    -------
    print statement with coherence values
    """
    
    topics = [["progression", "visibility"], ["city", "window"], ["continue", "snow"]]
    features = [['ohsttkoorj'], 
                ['progression', 'storm', 'qdosgepicb'], 
                ['snow', 'band', 'starting', 'pivot', 'avalon', 'heavy snow', 'continue', 'hour', 'wind', 'thqvt6xlwl'], 
                ['anytime', 'speculated', 'yesterday', 'severity', 'storm', 'headache', 'gonna', '4wziule2dc'], 
                ['eddiesheerr', 'eddie', 'twn', 'forecast', 'wind', 'starting', 'ease', 'hour', '00pm', 'accurate'], 
                ['picture', 'crazyy', 'complain', 'ottawa', 'winter'], 
                ['visibility', 'reduced', 'are', 'towngfw', 'hour', 'motorist', 'stayed', 'town', 'crew', 'bezohqsext'], 
                ['hiphops lit dope flow', 'gotdrip bing tiktok feelin', 'therealbigthump', 'sddgjenmwx'], 
                ['bonkers', 'snow', 'window', 'nkwgoh4ark'], 
                ['snow drift', 'taller', 'city', 'issued', 'garbage', 'bin', 'telling', 'set', 'stair', 'fhy7aovrzc']]
    for c in ['c_v', 'c_uci', 'c_npmi', 'u_mass']:
        coherence = get_coherence(features, topics, coherence_type = c)
        print(f"The coherence value of the model using the {c} metric is: {coherence}.")

# Use Gensim's CoherenceModel() to return model coherence:
# https://radimrehurek.com/gensim/models/coherencemodel.html
def get_coherence(features, topics, coherence_type = 'c_v'):
    dct = Dictionary(features)
    #topics = topics.values.tolist()

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

if __name__ == '__main__':
    main()
