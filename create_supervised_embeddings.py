import os
import pandas as pd
import pickle

from BertTM import *

def main():
    DATA_PATH = "../bert-classifier-pytorch/CrisisNLP_labeled_data_crowdflower"
    PATHS = ["2014_California_Earthquake", "2014_Chile_Earthquake_en", "2013_Pakistan_eq",
             "2014_Hurricane_Odile_Mexico_en","2014_India_floods", "2014_Pakistan_floods",
             "2014_Philippines_Typhoon_Hagupit_en","2015_Cyclone_Pam_en","2015_Nepal_Earthquake_en"]

    df = read_supervised_data(DATA_PATH, PATHS, "test")
    df = df[['tweet_text', 'choose_one_category']]
    df.columns=['text','prediction']
    data = df.text.tolist()
    pred = df.prediction.tolist()
    
    print("Initializing supervised model.")
    model, tokenizer = load_model(finetuned = True)
    print("Getting sentence embeddings for supervised model.")
    start_time = time.time()
    rows, attentions = get_embeddings(data, model, tokenizer)
    save_embeddings(data, pred, rows, attentions, "attentions_supervised_embeddings")
    print(f'Getting sentence embeddings for supervised test set took {round(time.time() - start_time, 2)} seconds.')
    

def read_supervised_data(data_path, paths, train_test):
    data = []
    counter = 0
    for path in paths:
        DIR = f"{data_path}/{path}"
        for filename in os.listdir(DIR):
            if filename.endswith(".csv") and train_test in filename:
                print(os.path.join(DIR, filename))
                with open(os.path.join(DIR, filename), 'r', encoding='utf8') as file:
                    df = pd.read_csv(file)
                    data.append(df)
                    counter += len(df["tweet_text"])
        print(f"Processed {counter} files.")
    print(f"Processed a total of {counter} files.")
    df_return = pd.concat(data)
    return(df_return)

def load_model(finetuned = True):
    if finetuned:
        output_dir = "../bert-classifier-pytorch/model_save_attention_1epoch"
        model = BertForSequenceClassificationOutputPooled.from_pretrained(output_dir,
                                                              output_attentions = True, 
                                                              output_hidden_states = True)
        tokenizer = BertTokenizer.from_pretrained(output_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', )
        model = BertForSequenceClassificationOutputPooled.from_pretrained('bert-base-uncased', 
                                                                      output_attentions=True, 
                                                                      output_hidden_states=True)
    return model, tokenizer

# Dump embeddings as pickle file
def save_embeddings(data, pred, rows, attentions, path):
    all_model_data = []
    for i in range(len(rows)):
        all_model_data.append((data[i], pred[i], attentions[i], rows[i]))
    pickle.dump(all_model_data, open(f"{path}.pkl", "wb" ))


if __name__ == '__main__':
    main()
