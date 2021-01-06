'''This module takes the names of two models as input and outputs the "agreement" of the two models. 
Agreement measures the extent to which the label outputs of two (unsupervised) models overlap.

Arguments
----------
    [0] : python
    [1] : agreement.py
    [2] : "bert", "btm" or "lda"
    [3] : "bert", "btm" or "lda"
'''

from collections import Counter
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import sys

def main():
    '''Check that correct sys.argvs are given and then compare models.'''
    
    models = sys.argv[1:3]
    cols = ['text', 'label']
    if set(['bert', 'lda']) == set(models):
        print('Comparing bert and lda.')
        bert_ft = get_bert_m()
        lda_df = pd.read_csv("data/lda_labels.csv", usecols=cols)
        avg_a = comp_bert_lda(bert_ft, lda_df, "ft", "lda")
    elif set(['bert', 'btm']) == set(models):
        print('Comparing bert and btm.')
        bert_ft = get_bert_m()
        btm_df = pd.read_csv("data/btm_labels.csv", usecols=cols)
        avg_a = comp_bert_lda(bert_ft,btm_df, "ft", "btm")
    elif set(['lda', 'btm']) == set(models):
        print('Comparing btm and lda.')
        lda_df = pd.read_csv("data/lda_labels.csv", usecols=cols)
        btm_df = pd.read_csv("data/btm_labels.csv", usecols=cols)
        avg_a = comp_2_models(lda_df,btm_df, "lda", "btm")
    else:
        print('Passed invalid model names. Use "bert", "lda" or "btm".')
    print(f'The average agreement between the two models is {avg_a}.')

def get_bert_m(model = "finetuned_sent_embeddings",
               PATH = "finetuned_sent_embeddings_ngram1_no_phrasing_stf-False_mdf-0.4_hashtags"):
    '''Load bert model and labels from corresponding topic model. Return dataframe with text and label columns.
    Parameters
    ----------
        model : str
        PATH : str
    Returns
    -------
    bert_df : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: text, dtype: object
            Name: label, dtype: int64
    '''
    all_model_data = pickle.load(open(f'bert_embedders/nlwx/{model}.pkl', 'rb'))
    texts, _, _, _ = zip(*all_model_data)
    labs = pickle.load(open(f'topic_models/nlwx/{PATH}/9/labels.pkl', 'rb'))
    bert_labs = list(map(str, labs)) 
    bert_df = pd.DataFrame(
        {'text': pd.Series(texts, dtype='object'),
         'label': pd.Series(bert_labs, dtype='int')
        })
    bert_df = bert_df.drop_duplicates()
    return(bert_df)

def comp_bert_lda(bert_df, lda_df, b_name, l_name):
    '''Load bert and lda models. Return float value of average model agreement.
    Parameters
    ----------
        bert_df : pandas.DataFrame
            Index: RangeIndex
            Columns:
                Name: text, dtype: object
                Name: label, dtype: int64
        lda_df : pandas.DataFrame
            Index: RangeIndex
            Columns:
                Name: text, dtype: object
                Name: label, dtype: int64
        b_name : str
        l_name : str
    Returns
    -------
    avg_a : float
    '''
    
    # Only include rows from BERT model included in the LDA output. 
    # Necessary due to slight differences in preprocessing and tokenization.
    bert_df = bert_df[bert_df.text.isin(lda_df.text)]
    avg_a = comp_2_models(bert_df, lda_df, b_name, l_name)
    return avg_a

def comp_2_models(m1, m2, m1_name, m2_name):
    '''Compare the agreement between two models. Saves outputs in outputs/agreement.
    Parameters
    ----------
        m1 : pandas.DataFrame
            Index: RangeIndex
            Columns:
                Name: text, dtype: object
                Name: label, dtype: int64
        m2 : pandas.DataFrame
            Index: RangeIndex
            Columns:
                Name: text, dtype: object
                Name: label, dtype: int64
        b_name : str
        l_name : str
    Returns
    -------
    avg_a : float
    
    '''
    m1 = pd.Series(list(map(str, m1.label)))
    m2 = pd.Series(list(map(str, m2.label)))

    ct = pd.crosstab(m1, m2, rownames=[m1_name], 
                     colnames=[m2_name], margins=False).apply(lambda r: round(r, 0))
    ct_norm_m1 = pd.crosstab(m1, m2, rownames=[m1_name], 
                                  colnames=[m2_name], margins=False, 
                                  normalize = 'index').apply(lambda r: round(r, 2))
    ct_norm_m2 = pd.crosstab(m1, m2, rownames=[m1_name], 
                           colnames=[m2_name], margins=False, 
                           normalize = 'columns').apply(lambda r: round(r, 2))

    ct.to_csv(f"outputs/agreement/{m1_name}_vs_{m2_name}.csv", index=True)
    ct_p = sns.heatmap(ct, cmap="YlGnBu", annot=True, cbar=False, fmt='d')
    ct_p.figure.savefig(f"outputs/agreement/{m1_name}_v_{m2_name}.png",dpi=300, bbox_inches="tight")
    plt.clf()
    
    a1 = get_agreement(ct_norm_m1, m1)
    a2 = get_agreement(ct_norm_m2, m2)
    avg_a = (a1+a2)/2
    return(avg_a)

def get_agreement(ct_norm, m):
    '''Check from embedding path whether topics include hashtags.
    Parameters
    ----------
        ct_norm : pandas.DataFrame
            Index: RangeIndex
            Columns:
                Name: 0, dtype: float64
                Name: 1, dtype: float64
                Name: 2, dtype: float64
                Name: 3, dtype: float64
                Name: 4, dtype: float64
                Name: 5, dtype: float64
                Name: 6, dtype: float64
                Name: 7, dtype: float64
                Name: 8, dtype: float64
        m : pandas.Series
    Returns
    -------
        agreement : float
    '''
    max_overlap = ct_norm.max()
    freqs = Counter(m)
    
    # Calculate agreement. See paper for details.
    agreement = sum([max_overlap[i]*(freqs[str(i)]/len(m)) for i in max_overlap.index])
    return(agreement)

if __name__ == '__main__':
    main()
