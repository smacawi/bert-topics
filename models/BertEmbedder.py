from sentence_transformers import SentenceTransformer, models, losses
from transformers import BertTokenizer, BertModel, BertPreTrainedModel

from models.BertForSequenceClassificationOutputPooled import BertForSequenceClassificationOutputPooled

import numpy as np
import pickle
import time
import torch
import torch.nn as nn

class BertEmbedder():
    def __init__(self, model_dir, sentence_bert=True):
        self.sentence_bert = sentence_bert
        self.model_dir = model_dir
        self._load_model()
    
    def _load_model(self):
        if self.sentence_bert:
            word_embedding_model = models.BERT(self.model_dir, max_seq_length = 240,)
            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            self.stmodel = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            self.model = BertForSequenceClassificationOutputPooled.from_pretrained(self.model_dir,
                                                                                   output_attentions = True, 
                                                                                   output_hidden_states = True)
        else:
            self.model = BertForSequenceClassificationOutputPooled.from_pretrained(self.model_dir,
                                                                              output_attentions = True, 
                                                                              output_hidden_states = True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
    
    # Produce document attentions and embeddings.
    # Embeddings returned using one of two methods:
    # 1) BERT pooled embedding layers per the pooler_output here: 
    #    https://huggingface.co/transformers/model_doc/bert.html
    # 2) Sentence transformers as outlined here: https://github.com/UKPLab/sentence-transformers
    def get_embeddings(self, data):
        self.embeddings, self.attentions = [], []
        start_time = time.time()
        for i in range(0, len(data)):
            if self.sentence_bert:
                self.embeddings.extend(self.stmodel.encode([data[i]]))
            else:
                self.embeddings.extend(vectorize([data[i]], self.model, self.tokenizer))
            self.attentions.extend(self._get_attention([data[i]]))
            if i % 50 == 0:
                print(f'Processed {(i)} rows in {round(time.time() - start_time, 2)} seconds.')
    
    # Dump embeddings as pickle file
    def save_outputs(self, texts, labels, outpath):
        all_data = []
        for i in range(len(self.embeddings)):
            all_data.append((texts[i], labels[i], self.attentions[i], self.embeddings[i]))
        pickle.dump(all_data, open(f"{outpath}.pkl", "wb" ))
    
    # Get attention weights from model. 
    # Currently implemented first layer and last layer. Last is current standard in academica.
    # Further implementation could include mean layer.
    def _get_attention(self, texts, method = 'last'):
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
        input_list, token_list = self._tokenize_for_tm(texts)
        sentence_attentions = []
        for idx, input_ids in enumerate(input_list):
            outputs = self.model(input_ids)
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
            sentence_attentions.append(self._merge_wordpiece_tokens(list(zip(token_list[idx], attn))))
        return(sentence_attentions)
    
    # Tokenize texts for vectorization and extracting attention weights.
    # Uses Huggingface tokenizers: 
    # https://huggingface.co/transformers/main_classes/tokenizer.html
    def _tokenize_for_tm(self, texts):
        input_list = []
        token_list = []
        cls_ = '[CLS]'
        sep_ = '[SEP]'
        for i, sent in enumerate(texts):
            inputs = self.tokenizer.encode_plus(texts[i], add_special_tokens=True)
            tokens = [cls_] + self.tokenizer.tokenize(texts[i]) + [sep_]
            input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
            input_list.append(input_ids)
            token_list.append(tokens)
        return(input_list, token_list)

    # Merge word piece tokens in order to get complete words for topic cluster components.
    # Implemented from block 14 of commit 9895ee0 at:
    # https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/topic-model/2.bert-topic.ipynb
    def _merge_wordpiece_tokens(self, paired_tokens, weighted = True):
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
    # CURRENTLY NOT WORKING!!!
    def _vectorize(texts, model, tokenizer):
        input_list, _ = tokenize_for_tm(texts, tokenizer)
        vectorized_sentences = []
        for idx, input_ids in enumerate(input_list):
            outputs = model(input_ids)
            # No labels given, so loss not in output and index is 3. 
            # With labels given, index is 4.
            vectorized_sentences.append(outputs[3][0].detach().numpy())
        return(vectorized_sentences)