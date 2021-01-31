'''Use coherence scores to select best performing BERT based topic models.
Compare coherence of BERT models to coherence of LDA and BTM models and output plots of comparison. 
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

cmap = ListedColormap(sns.color_palette())

def main():
    '''Load coherence scores for different model iterations and output comparison plots.'''
    c_metrics = ['c_v','c_npmi']
    topics = [5,9,10,15]
    dfs = get_dfs(c_metrics, topics)
    lda_df = pd.read_csv("data/lda_coherence.csv")
    output_full = concat_dfs(dfs, 
                             lda_df, 
                             path = "automated_coherence_PAPER.csv", 
                             save = False)
    output_plot(data = output_full, 
                y = 'Coherence_CV', 
                ylab = 'Coherence ($\mathregular{C_{v}}$)', 
                filename = 'CV_plot')
    output_plot(data = output_full, 
                y = 'Coherence_NPMI', 
                ylab = 'Coherence $\mathregular{NPMI}$)', 
                filename = 'NPMI_plot')

def concat_dfs(dfs, lda_df, path, save = True):
    '''Concatenate dfs with coherence scores and model parameters.
    Parameters
    ----------
    dfs : list of pandas.DataFrame
    lda_df : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: Model, dtype: object
            Name: topics, dtype: int64
            Name: Coherence_CV, dtype: float64
            Name: Coherence_NPMI, dtype: float64
            Name: Components, dtype: float64
    path : str
    save : bool
    
    Returns
    -------
    output_full : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: Components, dtype: object
            Name: Model, dtype: object
            Name: topics, dtype: int64
            Name: Coherence_CV, dtype: float64
            Name: Coherence_NPMI, dtype: float64
            Name: Coherence_NPMI_abs, dtype: float64
            Name: Model, components, dtype: object
    '''
    output = pd.concat(dfs)
    output['embeddings'] = output['embeddings'].replace(
        {'finetuned_sent_embeddings':'finetuned', 
         'base_sent_embeddings':'base'})
    output = output[output.embeddings == "finetuned"]
    output = output.replace(
        {'topics_attn':'attn', 
         'topics_tfidf':'tfidf',
         'topics_tfidf_attn':'tfidf-attn',
         'finetuned': 'FTE'})
    output = output.rename({'embeddings': 'Model', 
                            'components': 'Components'}, axis='columns')
    output_cv = output[output['ct'].str.contains('c_v')]
    output_npmi = output[output['ct'].str.contains('c_npmi')]
    output_cv = output_cv.rename({'coherence': 'Coherence_CV'}, axis='columns')
    output_cv['Coherence_NPMI'] = output_npmi['coherence'].tolist()
    output_plot = output_cv
    output_full = output_cv[['Components','Model','topics',
                             'Coherence_CV','Coherence_NPMI']].append(lda_df, ignore_index=True)
    output_full['Coherence_NPMI_abs'] = output_full.Coherence_NPMI.abs()
    if save == True:
        output_full.to_csv(path, index=False)
    output_full['Model, components'] = output_full["Model"] + ", " + output_full["Components"]
    output_full = output_full.replace({'LDA, NA': 'LDA',
                                      'BTM, NA': 'BTM'})
    return(output_full)
    
def output_plot(data, y, ylab, filename):
    '''Take df about BERT and LDA models as input and output plot comparing them.
    Parameters
    ----------
    data : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: Components, dtype: object
            Name: Model, dtype: object
            Name: topics, dtype: int64
            Name: Coherence_CV, dtype: float64
            Name: Coherence_NPMI, dtype: float64
            Name: Coherence_NPMI_abs, dtype: float64
            Name: Model, components, dtype: object
    y : str
    ylab : str
    filename : str
    '''
    sns.set(style="ticks", font_scale=1.2)
    p = sns.lineplot(data= data, 
                     x='topics', 
                     y='Coherence_CV',
                     #hue="Model", style="Components",
                     hue = 'Model, components',
                     style = 'Model, components',
                     palette='rocket',
                     markers=True)
    plt.xlabel('Number of topics')
    plt.ylabel(ylab)
    p.set_xticks(range(5,16))
    p.set_xticklabels(range(5,16))
    p.figure.savefig(f"outputs/best_tm/{filename}.png",dpi=300, bbox_inches="tight")
    plt.clf()

def get_dfs(c_metrics, topics, save = False):
    '''Load BERT coherence data for each topic number and model type.
    Parameters
    ----------
    c_metrics : list of str
    topics : list of int
    save : bool
    
    Returns
    -------
    dfs : list of pandas.DataFrame
    '''
    dfs = []
    for c in c_metrics:
        for t in topics:
            df = load_cscores(t = t, c_type = c)
            max_df = filter_values(df)
            if save == True:
                save_model(max_df, c_type = c)
            dfs.append(max_df)
    return dfs

def load_cscores(t, c_type = 'c_v'):
    '''Load coherence score data for BERT topic models for different topic numbers and coherence types.
    Parameters
    ----------
    t : int
    model_type : str
    
    Returns
    -------
    df : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: embeddings, dtype: object
            Name: model, dtype: object
            Name: components, dtype: object
            Name: topics, dtype: int64
            Name: ngrams_per_topic, dtype: int64
            Name: ct, dtype: object
            Name: coherence, components, dtype: float64
            Name: hashtags, dtype: object
            Name: phrasing, dtype: object
            Name: max_df, dtype: float64
            Name: stf, dtype: bool
            Name: ngrams, components, dtype: int64
    '''
    df = pd.read_csv(f"outputs/coherence/coherence_scores_nlwx_{t}-topics_{c_type}-coherence.csv")
    return df
    
def filter_values(df):
    '''For each topic number and coherence type, get the BERT topic model with the best coherence score.
    The best coherence score is the highest for u_mass and the lowest for c_uci, c_v and c_npmi.
    Filters out models with max_df >= 0.5, hashtags and unigrams in keywords.
    Parameters
    ----------
    df : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: embeddings, dtype: object
            Name: model, dtype: object
            Name: components, dtype: object
            Name: topics, dtype: int64
            Name: ngrams_per_topic, dtype: int64
            Name: ct, dtype: object
            Name: coherence, components, dtype: float64
            Name: hashtags, dtype: object
            Name: phrasing, dtype: object
            Name: max_df, dtype: float64
            Name: stf, dtype: bool
            Name: ngrams, components, dtype: int64

    Returns
    -------
    max_df : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: embeddings, dtype: object
            Name: model, dtype: object
            Name: components, dtype: object
            Name: topics, dtype: int64
            Name: ngrams_per_topic, dtype: int64
            Name: ct, dtype: object
            Name: coherence, components, dtype: float64
            Name: hashtags, dtype: object
            Name: phrasing, dtype: object
            Name: max_df, dtype: float64
            Name: stf, dtype: bool
            Name: ngrams, components, dtype: int64
    
    '''
    df = df[df['max_df']>0.5]
    df = df[df['ngrams']==1]
    df = df[df['hashtags']==True]
    df = df[df['embeddings'].isin(['base_sent_embeddings','finetuned_sent_embeddings'])]
    if df.ct.all() == 'c_v':
        max_df = df.loc[df.groupby(['embeddings','components'])['coherence'].idxmax()]
    if df.ct.all() == 'u_mass':
        max_df = df.loc[df.groupby(['embeddings','components'])['coherence'].idxmin()]
    if df.ct.all() == 'c_npmi' or df.ct.all() == 'c_uci':
        max_df = df.loc[df.groupby(['embeddings','components'])['coherence'].idxmin()]
    return max_df
    
def save_model(max_df, c_type = 'cv'):
    '''Save csv. with best performing BERT topic models. By default only saves for topics
    Parameters
    ----------
    max_df : pandas.DataFrame
        Index: RangeIndex
        Columns:
            Name: embeddings, dtype: object
            Name: model, dtype: object
            Name: components, dtype: object
            Name: topics, dtype: int64
            Name: ngrams_per_topic, dtype: int64
            Name: ct, dtype: object
            Name: coherence, components, dtype: float64
            Name: hashtags, dtype: object
            Name: phrasing, dtype: object
            Name: max_df, dtype: float64
            Name: stf, dtype: bool
            Name: ngrams, components, dtype: int64
    c_type : str
    '''
    outfolder = "best_tm"
    for index, row in max_df.iterrows():
        if row['phrasing']:
            p = 'phrasing'
        else:
            p = 'no_phrasing'
        if row['hashtags']:
            h = 'hashtags'
        else:
            h = 'no_hashtags'
        model = f"{row['embeddings']}_ngram{row['ngrams']}_{p}_stf-{row['stf']}_mdf-{row['max_df']}_{h}"
        c = row['components']
        fn = f"topic_models/nlwx/{model}/9/topics/{c}.csv"
        out_df = pd.read_csv(fn)
        out_df.to_csv(f"{outfolder}/{c_type}/{model}_{c}.csv", index=False)

if __name__ == '__main__':
    main()