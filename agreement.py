from collections import Counter
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns


def main():
    bert_ft = get_bert_m()
    #bert_b = get_bert_m(model = "base_sent_embeddings",
    #                    PATH = "base_sent_embeddings_ngram2_phrasing_stf-True_mdf-1.0_no_hashtags")
    lda_df = pd.read_csv("lda_labels.csv")
    btm_df = pd.read_csv("btm_labels.csv")
    
    avg_a = comp_bert_lda(bert_ft, lda_df, "ft", "lda")
    #avg_a = comp_2_models(lda_df,btm_df, "lda", "btm")
    print(avg_a)

def get_bert_m(model = "finetuned_sent_embeddings",
               PATH = "finetuned_sent_embeddings_ngram1_no_phrasing_stf-False_mdf-0.4_hashtags"):
    all_model_data = pickle.load(open(f'bert_embedders/nlwx/{model}.pkl', 'rb'))
    texts, _, _, _ = zip(*all_model_data)
    labs = pickle.load(open(f'topic_models/nlwx/{PATH}/9/labels.pkl', 'rb'))
    bert_labs = list(map(str, labs)) 
    bert_df = pd.DataFrame(
        {'text': texts,
         'label': bert_labs
        })
    bert_df = bert_df.drop_duplicates()
    return(bert_df)

def comp_bert_lda(bert_df, lda_df, b_name, l_name):
    bert_df = bert_df[bert_df.text.isin(lda_df.text)]
    avg_a = comp_2_models(bert_df, lda_df, b_name, l_name)
    return avg_a

def comp_2_models(m1, m2, m1_name, m2_name):
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

    ct.to_csv(f"{m1_name}_vs_{m2_name}.csv", index=True)
    ct_p = sns.heatmap(ct, cmap="YlGnBu", annot=True, cbar=False, fmt='d')
    ct_p.figure.savefig(f"{m1_name}_v_{m2_name}.png",dpi=300, bbox_inches="tight")
    plt.clf()
    
    a1 = get_agreement(ct_norm_m1, m1)
    a2 = get_agreement(ct_norm_m2, m2)
    avg_a = (a1+a2)/2
    return(avg_a)

def get_agreement(ct_norm, m):
    max_overlap = ct_norm.max()
    freqs = Counter(m)
    agreement = sum([max_overlap[i]*(freqs[str(i)]/len(m)) for i in max_overlap.index])
    return(agreement)

if __name__ == '__main__':
    main()
