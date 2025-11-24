from datasets import load_dataset
import csv

flores_de = load_dataset("gsarti/flores_101", "deu")
#flores_spa = load_dataset("gsarti/flores_101", "spa")
flores_eng = load_dataset("gsarti/flores_101", "eng")
flores_ces = load_dataset("gsarti/flores_101", "ces")
flores_ara = load_dataset("gsarti/flores_101", "ara")

import pandas as pd

flores_dev_df= pd.DataFrame([flores_de['dev'], flores_eng['dev'], flores_ces['dev'], flores_ara['dev']])

flores_devtest_df= pd.DataFrame([flores_de['devtest'], flores_eng['devtest'], flores_ces['devtest'], flores_ara['devtest']])

flores_df = pd.concat([flores_dev_df, flores_devtest_df], axis=1)

flores_df = flores_df.T

flores_df.columns = ['deu', 'eng', 'ces', 'ara']

flores_df= flores_df.reset_index()

flores_df['eng'][3]

import pandas as pd

flores_df = pd.read_csv('/content/flores_df.csv')

from collections import defaultdict, Counter

from transformers import PreTrainedTokenizerFast
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, use_auth_token= 'secret')

import ast

def tokenize_sentences(flores_df):
    token_lang_dict = defaultdict(list)  # Stores tokens mapped to languages

    # Iterate
    for _, row in flores_df.iterrows():
        for lang in flores_df.columns:
            entry = row[lang]
            if isinstance(entry, str):
              try:
                entry = ast.literal_eval(entry)  # Convert string to dictionary
              except (ValueError, SyntaxError):
                # Handle when the string is not a valid dictionary representation
                print(f"Skipping invalid dictionary for {lang}: {entry}")
            if isinstance(entry, dict) and 'sentence' in entry:  # Check validity
                sentence = entry['sentence']
                tokens = tokenizer.tokenize(sentence)  # Tokenize sentence
                for token in tokens:
                    token_lang_dict[token].append(lang)
                words = sentence.split()
                for word in words:
                  token_lang_dict[word].append(lang)  # Assign token to language

    return token_lang_dict

def compute_token_frequencies(token_lang_dict):
    token_freq = {}  # Stores most frequent language for each token

    for token, langs in token_lang_dict.items():
        lang_counts = Counter(langs)  # Count occurrence of languages
        most_common_lang = lang_counts.most_common(1)[0][0]  # Get most frequent language
        token_freq[token] = most_common_lang

    return token_freq

# Assuming flores_df is already loaded
token_lang_dict = tokenize_sentences(flores_df)
token_freq_dict = compute_token_frequencies(token_lang_dict)

#Sanity check
token = "Munich"
token = tokenizer.tokenize(token)
token = token[0]
print(f"Most common language for '{token}':", token_freq_dict.get(token))
print(token)

with open('token_lang_frequencies_with_words.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Token', 'Most Frequent Language'])  # Header
    for token, lang in token_freq_dict.items():
        writer.writerow([token, lang])

for _, row in flores_df.iterrows():
        for lang in flores_df.columns:
            entry = row[lang]
            if isinstance(entry, dict) and 'sentence' in entry:  # Check validity
                sentence = entry['sentence']
                print(sentence)

len(token_freq_dict)
