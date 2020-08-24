from collections import Counter
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns

BASE_URL = "topic_models/supervised_sent_embeddings_hashtags/9/"
LABELS_PATH = f"{BASE_URL}labels.pkl"
PREDS_PATH = f"{BASE_URL}preds.pkl"

superv = pickle.load(open(PREDS_PATH, 'rb'))
usuperv = pickle.load(open(LABELS_PATH, 'rb'))
usuperv = list(map(str, usuperv)) 

cm = confusion_matrix(superv, usuperv, normalize='all')

cluster = pd.Series(usuperv)
pred = pd.Series(superv)

ct = pd.crosstab(pred, cluster, rownames=['Prediction'], colnames=['Cluster'], margins=False).apply(lambda r: round(r, 0))
ct_norm_cluster = pd.crosstab(pred, cluster, rownames=['Prediction'], 
                              colnames=['Cluster'], margins=False, 
                              normalize = 'columns').apply(lambda r: round(r, 2))
ct_norm_pred = pd.crosstab(pred, cluster, rownames=['Prediction'], 
                           colnames=['Cluster'], margins=False, 
                           normalize = 'index').apply(lambda r: round(r, 2))

ct.to_csv("label_pred_ct.csv", index=True)
ct_norm_cluster.to_csv("ct_norm_cluster.csv", index=True)
ct_norm_pred.to_csv("ct_norm_pred.csv", index=True)

ct_p = sns.heatmap(ct, cmap="YlGnBu", annot=True, cbar=False, fmt='d')
ct_p.figure.savefig("ct_p.png",dpi=300, bbox_inches = "tight")
plt.clf()

ct_norm_cluster_p = sns.heatmap(ct_norm_cluster,cmap="YlGnBu", annot=True, cbar=False)
ct_norm_cluster_p.figure.savefig("ct_norm_cluster_p.png",dpi=300, bbox_inches = "tight")
plt.clf()

ct_norm_pred_p = sns.heatmap(ct_norm_pred,cmap="YlGnBu", annot=True, cbar=False)
ct_norm_pred_p.figure.savefig("ct_norm_pred_p.png",dpi=300, bbox_inches = "tight")

max_overlap = ct_norm_cluster.max()
freqs = Counter(cluster)
accuracy = sum([max_overlap[i]*(freqs[i]/len(cluster)) for i in max_overlap.index])

print(f"The overall accuracy of the unsupervised classifier is: {accuracy}")