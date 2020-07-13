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