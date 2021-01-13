from BertForSequenceClassificationOutputPooled import *
from collections import Counter
from datetime import datetime
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob, Word
from transformers import AdamW, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel

import json
import math
import numpy as np
import pandas as pd
import re
import time
import torch
import torch.nn as nn

# Tokenize texts for vectorization and extracting attention weights.
# Uses Huggingface tokenizers: 
# https://huggingface.co/transformers/main_classes/tokenizer.html
def tokenize_for_tm(texts, tokenizer):
    input_list = []
    token_list = []
    cls_ = '[CLS]'
    sep_ = '[SEP]'
    for i, sent in enumerate(texts):
        inputs = tokenizer.encode_plus(texts[i], add_special_tokens=True)
        tokens = [cls_] + tokenizer.tokenize(texts[i]) + [sep_]
        input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
        input_list.append(input_ids)
        token_list.append(tokens)
    return(input_list, token_list)

# Merge word piece tokens in order to get complete words for topic cluster components.
# Implemented from block 14 of commit 9895ee0 at:
# https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
def merge_wordpiece_tokens(paired_tokens, weighted = True):
    new_paired_tokens = []
    n_tokens = len(paired_tokens)
    i = 0
    while i < n_tokens:
        current_token, current_weight = paired_tokens[i]
        if current_token.startswith('##'):
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while current_token.startswith('##'):
                merged_token = merged_token + current_token.replace('##', '')
                merged_weight.append(current_weight)
                i = i + 1
                current_token, current_weight = paired_tokens[i]
            merged_weight = np.mean(merged_weight)
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1

    words = [
        i[0]
        for i in new_paired_tokens
        if i[0] not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    weights = [
        i[1]
        for i in new_paired_tokens
        if i[0] not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    if weighted:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    return list(zip(words, weights))

# Vectorize input documents using pooling_input similar to "pooler_output" in:
# https://huggingface.co/transformers/model_doc/bert.html
def vectorize(texts, model, tokenizer):
    input_list, _ = tokenize_for_tm(texts, tokenizer)
    vectorized_sentences = []
    
    for idx, input_ids in enumerate(input_list):
        outputs = model(input_ids)
        # No labels given, so loss not in output and index is 3. 
        # With labels given, index is 4.
        vectorized_sentences.append(outputs[3][0].detach().numpy())
    return(vectorized_sentences)

# Get attention weights from model. 
# Currently implemented first layer and last layer. Last is current standard in academica.
# Further implementation could include mean layer.
def get_attention(texts, model, tokenizer, method = 'last'):
    """
    ATTENTION SHAPE GUIDE:
    outputs[object_in_output][tuple_length = nr of hidden layers][batch_size][num_heads][sequence_length][sequence_length]
    object_in_output = 4 -> want: 3
    tuple_length = number of layers = 12, but depends on model -> want: all -> later first (0) or last (-1)
    batch_size = 1, but depends on model -> want: all
    num_heads =  12, but depends on model -> want: all
    sequence_length = depends on model -> want: 0 to get CLS
    sequence_length = depends on model -> want: ALL to see what CLS attends to
    """
    input_list, token_list = tokenize_for_tm(texts, tokenizer)
    sentence_attentions = []
    for idx, input_ids in enumerate(input_list):
        outputs = model(input_ids)
        sentence_attention = torch.stack(outputs[2])[:,:,:,:1,:]
        if method == 'first':
            cls_attn = sentence_attention[:1][:][:][:][:]
        if method == 'last':
            cls_attn = sentence_attention[-1:,:,:,:,:]

        cls_attn = torch.mean(cls_attn, axis = 2)
        total_weights = torch.sum(cls_attn)
        attn = cls_attn / total_weights
        attn = attn.flatten()
        attn = list(attn.detach().numpy())
        sentence_attentions.append(merge_wordpiece_tokens(list(zip(token_list[idx], attn))))
    return(sentence_attentions)

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
    
# Loop through attention and token pairs to generate ngrams for topic cluster components.
# Implemented from block 14 of commit 9895ee0 at:
# https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
def generate_ngram(seq, ngram = (1, 3)):
    g = []
    for i in range(ngram[0], ngram[-1] + 1):
        g.extend(list(ngrams_generator(seq, i)))
    return g


# Pad sequence helper funtion for ngram generator.
# Implemented from block 14 of commit 9895ee0 at:
# https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
def _pad_sequence(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

# Use attention and token pairs to generate ngrams for topic cluster components.
# Implemented from block 14 of commit 9895ee0 at:
# https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
def ngrams_generator(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
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
    sequence = _pad_sequence(
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

# Produce document attentions and embeddings.
# Embeddings returned using one of two methods:
# 1) BERT pooled embedding layers per the pooler_output here: https://huggingface.co/transformers/model_doc/bert.html
# 2) Sentence transformers as outlined here: https://github.com/UKPLab/sentence-transformers
def get_embeddings(data, model, tokenizer, pooled = False):
    rows, attentions = [], []
    start_time = time.time()
    if not pooled:
        output_dir = "../bert-classifier-pytorch/model_save_attention_1epoch"
        word_embedding_model = models.BERT(output_dir, max_seq_length = 240,)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    for i in range(0, len(data)):
        if pooled:
            rows.extend(vectorize([data[i]], model, tokenizer))
        else:
            rows.extend(st_model.encode([data[i]]))
        attentions.extend(get_attention([data[i]], model, tokenizer))
        if i % 50 == 0:
            print(f'Processed {(i)} rows in {round(time.time() - start_time, 2)} seconds.')
    return rows, attentions

# Get count for number of documents per topic cluster.
def get_label_counts(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print("The number of texts per label are:")
    print(dict(zip(unique, counts)))
    
# Get topic clusters using kmeans from sklearn.
def get_clusters(rows, n_clusters):
    print("Fitting kmeans model.")
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(rows)
    labels = kmeans.labels_
    get_label_counts(labels)
    return labels, kmeans

# Filter attentions and wordpiece tokens for stopwords
def filter_data(attentions, stopwords, labels):
    filtered_a, filtered_t, filtered_l = [], [], []
    url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    print("Filtering attentions.")
    for idx, a in enumerate(attentions):
        f = [(i[0].lower(), i[1]) for i in a]
        f = [(Word(i[0]).lemmatize(), i[1]) 
             for i in f if (i[0] not in stopwords) 
             and (not re.match(url_re, i[0]))
             and (i[0].find('snowmageddon2020') == -1)]
        f_txt = [re.sub('[^a-zA-Z]+', '', w[0]) for w in f]
        f = [(re.sub('[^a-zA-Z]+', '', w[0]), w[1]) for w in f]
        if len(f) > 1:
            filtered_a.append(f)
            filtered_t.append(f_txt)
            filtered_l.append(labels[idx])
    return filtered_a, filtered_t, filtered_l

def get_frequent_vocab(corpus, threshold=10):
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

def remove_ifreq_words(features, vocab_threshold = 10):
    texts = [word for words in features for word in words]
    vocab = get_frequent_vocab(texts, threshold = vocab_threshold)
    updated_f = []
    print(len(vocab))
    for f in features:
        updated_f.append([word for word in f if word in vocab and len(word) > 0])
    return updated_f

# Determine the words and ngrams beloning to each topic cluster.
# Major modification on code in block 27 of commit 9895ee0  at:
# https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
def determine_cluster_components(filtered_l, filtered_a, ngram, features):
    print("""
Determining cluster components. This will take a while. 
Progress will be printed for every 500th processed property.
    """)
    components = {}
    words_label = {}
    start_time = time.time()
    for idx, label in enumerate(filtered_l):
        if label not in components:
            components[label] = {}
            words_label[label] = []
        else:
            f = generate_ngram(filtered_a[idx], ngram)
            for w in f:
                word = ' '.join([r[0] for r in w])
                score = np.mean([r[1] for r in w])
                if word in features[idx]:
                    if word in components[label]:
                        components[label][word] += score
                    else:
                        components[label][word] = score
                    words_label[label].append(word)
        if (idx + 1) % 5000 == 0:
            print(f'Processed {(idx + 1)} texts in {round(time.time() - start_time, 2)} seconds.')

    print(f"Finished determining a total of {idx + 1} cluster components. Total time {round(time.time() - start_time, 2)} seconds.")
    return(components, words_label)

# Dummy function for creating tf-idf vectorizer in sklearn.
# From http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/
def dummy_fun(doc):
    return doc

# Create tf-idf weights for topic clusters using Sklearn.
def tf_icf(words_label, n_topics, max_df = 1.0, stf = False):
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        max_df = max_df,
        sublinear_tf = stf)
    
    tf_idf_corpus = [[item for item in words_label[key]] for key in range(n_topics)]
    transformed = tfidf_vectorizer.fit_transform(tf_idf_corpus)
    index_value={i[1]:i[0] for i in tfidf_vectorizer.vocabulary_.items()}
    fully_indexed = []
    for row in transformed:
        fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})
    return(fully_indexed)

# Get topic cluster component weights using attention, tf-idf and a mixture of both.
def get_tfidf_components(components, tfidf_indexed):
    components_tfidf_attn = {}
    components_tfidf = {}
    for k1 in components:
        components_tfidf_attn[k1] = {}
        components_tfidf[k1] = {}
        for k2 in components[k1]:
            try:
                components_tfidf_attn[k1][k2] = tfidf_indexed[k1][k2] * components[k1][k2]
                components_tfidf[k1][k2] = tfidf_indexed[k1][k2]
            except:
                continue
    return(components_tfidf, components_tfidf_attn)

# Generate phrases for topic cluster components using Gensim Phrases() and Phraser() functions:
# https://radimrehurek.com/gensim/models/phrases.html
def get_phrases(filtered_t, min_count=5, threshold=100):
    bigram = Phrases(filtered_t, min_count=min_count, threshold = threshold) # higher threshold fewer phrases.
    trigram = Phrases(bigram[filtered_t])  

    # 'Phraser' is a wrapper that makes 'Phrases' run faster
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)

    phrased_bi = [b for b in bigram[filtered_t]]
    phrased_tri = [t for t in trigram[[b for b in bigram[filtered_t]]]]
    features = [[w.replace('_', ' ') for w in sublist] for sublist in phrased_bi]
    return(features)

# Import stopwords and hashtags to drop when deternmining topic cluster components.
# These stopwords will _not_ be dropped from the actual BERT models when determining topic clusters.
def get_stopwords(hashtags = [], filename = 'stopwords-en.json'):
    with open(filename) as fopen:
        stopwords = json.load(fopen)

    stopwords.extend(['#', '@', '…', "'", "’", "[unk]", "\"", ";", 
                      "*", "_", "amp", "&", "“", "”"] + hashtags)
    return(stopwords)