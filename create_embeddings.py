import pandas as pd
import pickle

from BertTM import *

def main():
    df = pd.read_csv("nlwx_2020_hashtags_no_rt_predictions.csv")
    data = df['text']
    print(len(data))
    
    # 1. FINETUNED MODELS
    print("1. Initializing finetuned model.")
    model, tokenizer = load_model()
    
    ## 1A. With sentence embeddings
    #print("1A. Getting sentence embeddings for finetuned model.")
    #start_time = time.time()
    #rows, attentions = get_embeddings(data, model, tokenizer)
    #save_embeddings(data, df, rows, attentions, "attentions_finetuned_sent_embeddings")
    #print(f'Getting sentence embeddings for finetunedtook {round(time.time() - start_time, 2)} seconds.')
    
    ## 1B. With pooled embeddings
    #print("1B. Getting pooled embeddings for finetuned model.")
    #start_time = time.time()
    #rows, attentions = get_embeddings(data, model, tokenizer, pooled = True)
    #save_embeddings(data, df, rows, attentions, "attentions_finetuned_pooled")
    #print(f'Getting pooled embeddings for finetuned took {round(time.time() - start_time, 2)} seconds.')
    
    # 2. BASELINE MODELS
    print("2. Initializing baseline model.")
    model, tokenizer = load_model(finetuned = False)
    
    ## 2A. With sentence embeddings
    print("2A. Getting sentence embeddings for baseline model.")
    start_time = time.time()
    rows, attentions = get_embeddings(data, model, tokenizer)
    save_embeddings(data, df, rows, attentions, "attentions_base_sent_embeddings")
    print(f'Getting sentence embeddings for baseline took {round(time.time() - start_time, 2)} seconds.')
    
    # 2B. With pooled embeddings
    #print("2B. Getting pooled embeddings for baseline model.")
    #start_time = time.time()
    #rows, attentions = get_embeddings(data, model, tokenizer, pooled = True)
    #save_embeddings(data, df, rows, attentions, "attentions_base_pooled")
    #print(f'Getting pooled embeddings for baseline took {round(time.time() - start_time, 2)} seconds.')

# 
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
def save_embeddings(data, df, rows, attentions, path):
    all_model_data = []
    for i in range(len(rows)):
        all_model_data.append((data[i], df.prediction[i], attentions[i], rows[i]))
    pickle.dump(all_model_data, open(f"{path}.pkl", "wb" ))

if __name__ == '__main__':
    main()
