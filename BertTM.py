from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AdamW, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Redundant function but saving for now
def bert_tokenization(tokenizer, texts,cls = '[CLS]', sep = '[SEP]'):

    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        inputs = tokenizer.encode_plus(text, add_special_tokens=True)
        tokens = tokenizer.tokenize(text)
        tokens = [cls] + tokens + [sep]
        token_type_ids = inputs['token_type_ids']
        input_id = torch.tensor(inputs['input_ids']).unsqueeze(0)
        attention_mask = inputs['attention_mask']

        input_ids.append(input_id)
        input_masks.append(attention_mask)
        segment_ids.append(token_type_ids)
        s_tokens.append(tokens)

    maxlen = max([len(i) for i in input_ids])
    #input_ids = padding_sequence(input_ids, maxlen)
    #input_masks = padding_sequence(input_masks, maxlen)
    #segment_ids = padding_sequence(segment_ids, maxlen)

    return input_ids, input_masks, segment_ids, s_tokens

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

def vectorize(texts, model, tokenizer):
    input_list, _ = tokenize_for_tm(texts, tokenizer)
    vectorized_sentences = []
    
    for idx, input_ids in enumerate(input_list):
        outputs = model(input_ids)
        # No labels given, so loss not in output and index is 3. 
        # With labels given, index is 4.
        vectorized_sentences.append(outputs[3][0]) 
    return(vectorized_sentences)

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

def topics_df(topics, components, n_words = 20):
    df = {}
    for i in range(topics):
        words = sorted(components[i], key=components[i].get, reverse=True)[:n_words]
        df['topic %d' % (i)] = words
    return pd.DataFrame.from_dict(df)
    
def generate_ngram(seq, ngram = (1, 3)):
    g = []
    for i in range(ngram[0], ngram[-1] + 1):
        g.extend(list(ngrams_generator(seq, i)))
    return g

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

def preprocess(df,
               tweet_col='text',
               to_lower = True):
    
    df_copy = df.copy()

    # drop rows with empty values
    df_copy.dropna(how='all', inplace=True)
    # drop rows with identical text
    df_copy.drop_duplicates(subset = 'text', inplace = True)

    # lower the tweets
    if to_lower:
        df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str.lower()
    else:
        df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str

    # filter out stop words and URLs
    stopwords = ['&amp;', 'rt','th', 'co', 're', 've', 'kim', 'daca','#nlwhiteout', 
                 '#nlweather', 'newfoundland', '#nlblizzard2020', '#nlstm2020', '#snowmaggedon2020', 
                 '#stmageddon2020', '#stormageddon2020','#snowpocalypse2020', '#snowmageddon', '#nlstm', 
                 '#nlwx', '#nlblizzard', 'nlwhiteout', 'nlweather', 'newfoundland', 'nlblizzard2020', 'nlstm2020',
                 'snowmaggedon2020', 'stmageddon2020', 'stormageddon2020','snowpocalypse2020', 'snowmageddon', 
                 'nlstm', 'nlwx', 'nlblizzard', '#', '@', '…', "'", "’", "[UNK]", "\"", ";", "*", "_", "amp", "&"]
    url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    # to remove mentions add ?:\@| at the beginning
    df_copy['preprocessed_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: ' '.join(
        [re.sub(r"\b@\w+", "", word) for word in row.split() if (not word in stopwords) and (not re.match(url_re, word))]))
    # tokenize the tweets
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
    df_copy['tokenized_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: ' '.join(tokenizer.tokenize(row)))

    #remove tweets with length less than two
    df_copy = df_copy[df_copy['tokenized_' + tweet_col].map(len) >= 2]

    return(df_copy.reset_index())

# From http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/
def dummy_fun(doc):
    return doc

def tf_icf(words_label):
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None) 

    tf_idf_corpus = [[item for item in words_label[key]] for key in range(0,10)]
    transformed = tfidf_vectorizer.fit_transform(tf_idf_corpus)
    
    index_value={i[1]:i[0] for i in tfidf_vectorizer.vocabulary_.items()}
    fully_indexed = []
    for row in transformed:
        fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})
    return(fully_indexed)