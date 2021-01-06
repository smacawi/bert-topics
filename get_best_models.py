'''Use the Gensim module CoherenceModel() to calculate coherence for different models.
The coherence types used by default are c_v and c_npmi.
Outputs a csv file for each coherence type and for each topic number (5, 9, 10, 15).

See Gensim documentation for more details: https://radimrehurek.com/gensim/models/coherencemodel.html
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

cmap = ListedColormap(sns.color_palette())

def main():
    models = ['c_v','c_npmi']
    topics = [5,9,10,15]
    dfs = []
    for m in models:
        for t in topics:
            df = load_model(t = t, model_type = m)
            max_df = filter_values(df)
            #save_model(max_df, c_type = m)
            dfs.append(max_df)
    output = pd.concat(dfs)
    output['embeddings'] = output['embeddings'].replace(
        {'finetuned_sent_embeddings':'finetuned', 
         'base_sent_embeddings':'base'})
    output = output[output.embeddings == "finetuned"]
    lda_d = {'Model': ['LDA','BTM','LDA','BTM','LDA','BTM','LDA','BTM'],
              'topics': [5,5,9,9,10,10,15,15],
              'Coherence_CV':[0.2224171899548364,0.3757168285376632,0.3067994006844505,0.4794707561935,
                          0.3378062584495053,0.4515842003382982,0.46649009416555853,0.4382210025794913],
              'Coherence_NPMI':[-0.08974857376986704,0.10723389644054618,-0.1336361200421473,0.14771913134048534,
                               -0.14914148796623572,0.11959788877719732,-0.27850208863076364,0.09991915217690188],
              'Components':['NA']*8}

    lda_df = pd.DataFrame(lda_d, columns = ['Model', 'topics','Coherence_CV',
                                            'Coherence_NPMI','Components'])
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
    print(output_npmi['coherence'])
    output_cv['Coherence_NPMI'] = output_npmi['coherence'].tolist()
    output_plot = output_cv
    output_full = output_cv[['Components','Model','topics',
                             'Coherence_CV','Coherence_NPMI']].append(lda_df, ignore_index=True)
    output_full['Coherence_NPMI_abs'] = output_full.Coherence_NPMI.abs()
    output_full.to_csv("automated_coherence_PAPER.csv", index=False)
    output_full['Model, components'] = output_full["Model"] + ", " + output_full["Components"]
    output_full = output_full.replace({'LDA, NA': 'LDA',
                                      'BTM, NA': 'BTM'})
    print(output_full)

    output_plot(data = output_full, y = 'Coherence_CV', ylab = 'Coherence ($\mathregular{C_{v}}$)', filename = 'CV_plot.png')
    output_plot(data = output_full, y = 'Coherence_NPMI', ylab = 'Coherence $\mathregular{NPMI}$)', filename = 'NPMI_plot')

def output_plot(data, y, ylab, filename):
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
    p.figure.savefig(f"img/CV_plot.png",dpi=300, bbox_inches="tight")
    plt.clf()
    
def load_model(t, model_type = 'c_v'):
    df = pd.read_csv(f"outputs/coherence/coherence_scores_nlwx_{t}-topics_{model_type}-coherence.csv")
    return df
    
def filter_values(df):
    df = df[df['max_df']>0.5]
    df = df[df['ngrams']==1]
    df = df[df['hashtags']=='yes']
    df = df[df['embeddings'].isin(['base_sent_embeddings','finetuned_sent_embeddings'])]
    if df.ct.all() == 'c_v':
        max_df = df.loc[df.groupby(['embeddings','components'])['coherence'].idxmax()]
        #max_df = df.groupby(['embeddings','components'])['coherence'].transform(max) == df['coherence']
    if df.ct.all() == 'u_mass':
        #max_df = df.iloc[(df['coherence']).abs().argsort()[:2]]
        max_df = df.loc[df.groupby(['embeddings','components'])['coherence'].idxmin()]
    if df.ct.all() == 'c_npmi' or df.ct.all() == 'c_uci':
        max_df = df.loc[df.groupby(['embeddings','components'])['coherence'].idxmin()]
    return max_df
    
def save_model(max_df, c_type = 'cv'):
    outfolder = "best_tm"
    for index, row in max_df.iterrows():
        if row['phrasing'] == 'yes':
            p = 'phrasing'
        else:
            p = 'no_phrasing'

        if row['hashtags'] == 'yes':
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