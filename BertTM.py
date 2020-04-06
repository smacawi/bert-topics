from transformers import AdamW, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel

import numpy as np
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