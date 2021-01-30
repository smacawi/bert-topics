from collections import Counter
from gensim.models.phrases import Phrases, Phraser
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob, Word

import numpy as np
import re
import time

URL_RE = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'

class BertTopicModel():
    def __init__(self, texts, attentions, embeddings, n_clusters):
        self.texts = texts
        self.attentions = attentions
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        self.filtered_a = []
        self.filtered_t = [] 
        self.filtered_l = []
        
    def get_clusters(self):
        print("Fitting kmeans model.")
        kmeans = KMeans(n_clusters = self.n_clusters, random_state = 0).fit(self.embeddings)
        self.labels = kmeans.labels_
        self._get_label_counts()
        return self.labels
    
    def get_features(self, stopwords, phrasing = False,min_count=5, threshold=100):
        self.filter_data(stopwords)
        if phrasing == True:
            self._get_phrases(min_count=min_count, threshold=threshold)
        else:
            self.features = self.filtered_t
        self._remove_ifreq_words()
        return self.features
        
    def filter_data(self, stopwords):
        url_re = URL_RE
        print("Filtering attentions.")
        for idx, a in enumerate(self.attentions):
            f = [(i[0].lower(), i[1]) for i in a]
            f = [(Word(i[0]).lemmatize(), i[1]) 
                 for i in f if (i[0] not in stopwords) 
                 and (not re.match(url_re, i[0]))
                 and (i[0].find('snowmageddon2020') == -1)]
            f_txt = [re.sub('[^a-zA-Z]+', '', w[0]) for w in f]
            f = [(re.sub('[^a-zA-Z]+', '', w[0]), w[1]) for w in f]
            if len(f) > 1:
                self.filtered_a.append(f)
                self.filtered_t.append(f_txt)
                self.filtered_l.append(self.labels[idx])
                
    def determine_cluster_components(self, ngram):
        print("""
    Determining cluster components. This will take a while. 
    Progress will be printed for every 500th processed property.
        """)
        components = {}
        words_label = {}
        start_time = time.time()
        for idx, label in enumerate(self.filtered_l):
            if label not in components:
                components[label] = {}
                words_label[label] = []
            else:
                f = self._generate_ngram(self.filtered_a[idx], ngram)
                for w in f:
                    word = ' '.join([r[0] for r in w])
                    score = np.mean([r[1] for r in w])
                    if word in self.features[idx]:
                        if word in components[label]:
                            components[label][word] += score
                        else:
                            components[label][word] = score
                        words_label[label].append(word)
            if (idx + 1) % 5000 == 0:
                print(f'Processed {(idx + 1)} texts in {round(time.time() - start_time, 2)} seconds.')

        print(f"Finished determining a total of {idx + 1} cluster components.\
        Total time {round(time.time() - start_time, 2)} seconds.")
        self.components = components 
        self.words_label = words_label
        return self.components, self.words_label
    
    def get_tfidf_components(self, max_df = 1.0, stf = False):
        tfidf_indexed = self._tf_icf(max_df, stf)
        components_tfidf_attn = {}
        components_tfidf = {}
        for k1 in self.components:
            components_tfidf_attn[k1] = {}
            components_tfidf[k1] = {}
            for k2 in self.components[k1]:
                try:
                    components_tfidf_attn[k1][k2] = tfidf_indexed[k1][k2] * self.components[k1][k2]
                    components_tfidf[k1][k2] = tfidf_indexed[k1][k2]
                except:
                    continue
        self.components_tfidf = components_tfidf
        self.components_tfidf_attn = components_tfidf_attn
        return self.components_tfidf,self.components_tfidf_attn
    
    def _get_label_counts(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        print("The number of texts per label are:")
        print(dict(zip(unique, counts)))
        
    # Generate phrases for topic cluster components using Gensim Phrases() and Phraser() functions:
    # https://radimrehurek.com/gensim/models/phrases.html
    def _get_phrases(self, min_count=5, threshold=100):
        bigram = Phrases(self.filtered_t, min_count=min_count, threshold = threshold) # higher threshold fewer phrases.
        trigram = Phrases(bigram[self.filtered_t])  

        # 'Phraser' is a wrapper that makes 'Phrases' run faster
        bigram_phraser = Phraser(bigram)
        trigram_phraser = Phraser(trigram)

        phrased_bi = [b for b in bigram[self.filtered_t]]
        phrased_tri = [t for t in trigram[[b for b in bigram[self.filtered_t]]]]
        self.features = [[w.replace('_', ' ') for w in sublist] for sublist in phrased_bi]
        
    def _remove_ifreq_words(self, vocab_threshold = 10):
        texts = [word for words in self.features for word in words]
        vocab = self._get_frequent_vocab(texts, threshold = vocab_threshold)
        updated_f = []
        print(len(vocab))
        for f in self.features:
            updated_f.append([word for word in f if word in vocab and len(word) > 0])
        self.features = updated_f
    
    def _get_frequent_vocab(self, corpus, threshold=10):
        '''
        Gets words whose frequency exceeds that of the threshold
        in a given corpus.
        :param corpus: list of tokenized words
        :param threshold:
        :return: list of words with higher frequency than threshold
        '''
        freq = Counter(corpus)
        filtered = [word for word, count in freq.items() if count >= threshold]
        return filtered
    
    def _generate_ngram(self, 
                        seq, 
                        ngram = (1, 3)):
        g = []
        for i in range(ngram[0], ngram[-1] + 1):
            g.extend(list(self._ngrams_generator(seq, i)))
        return g
    
    def _ngrams_generator(
        self,
        sequence,
        n,
        pad_left = False,
        pad_right = False,
        left_pad_symbol = None,
        right_pad_symbol = None):
        """
        generate ngrams.

        Parameters
        ----------
        sequence : list of str
            list of tokenize words.
        n : int
            ngram size

        Returns
        -------
        ngram: list
        """
        sequence = self._pad_sequence(
            sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
        )

        history = []
        while n > 1:
            try:
                next_item = next(sequence)
            except StopIteration:
                return
            history.append(next_item)
            n -= 1
        for item in sequence:
            history.append(item)
            yield tuple(history)
            del history[0]
            
    # Pad sequence helper funtion for ngram generator.
    # Implemented from block 14 of commit 9895ee0 at:
    # https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
    def _pad_sequence(
        self,
        sequence,
        n,
        pad_left = False,
        pad_right = False,
        left_pad_symbol = None,
        right_pad_symbol = None):
        
        sequence = iter(sequence)
        if pad_left:
            sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
        if pad_right:
            sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
        return sequence
    
    def _tf_icf(self, max_df, stf):
        def dummy_fun(doc):
            return doc
        
        tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            token_pattern=None,
            max_df = max_df,
            sublinear_tf = stf)

        tf_idf_corpus = [[item for item in self.words_label[key]] for key in range(self.n_clusters)]
        transformed = tfidf_vectorizer.fit_transform(tf_idf_corpus)
        index_value={i[1]:i[0] for i in tfidf_vectorizer.vocabulary_.items()}
        fully_indexed = []
        for row in transformed:
            fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})
        return(fully_indexed)
    

