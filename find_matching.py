# find_matching.py
# Python script for finding similar rare words across two corpuses
# © 2023 Archie McKenzie, MIT License

# For each pair, tokenizes using tiktoken
# Creates a frequency table of tokens used in each corpus
# Then, rearranges sentences by the rarity of their matching words

# ------- IMPORTS ------- #

# Import natural language toolkit
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

# Import tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Import counter for the frequency tables
from collections import Counter

# Import json for exporting output
import json

# ------- EDITABLE VARIABLES ------- #
input_path_1 = 'input/dial_of_princes.txt'
input_path_2 = 'input/wits_misery.txt'
unsorted_similarities_path = 'output/output.json'
output_path = 'output/matching.json' # sorted

# ------- OTHER FUNCTIONS ------- #

# Takes in a passage, tokenizes it, and returns a frequency table for each token
def make_token_freq_table(text):
    tokenized_text = enc.encode(text)
    return len(tokenized_text), Counter(tokenized_text) # construct frequency table using Counter

# Split any given string into sentences
# Return an array of strings
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Tokenize each of the strings in a given array of strings
# Return an array of integers (tokens)
def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(enc.encode(sentence))
    return tokenized_sentences

# Sorts pairs into groups by their rarest words, then ranks them based on semantic similarity
def sort_by_matching_word_rarity_then_similarity(pairs):
    # Sort by matchingWordRarity in descending order
    sorted_pairs = sorted(pairs, key=lambda x: x['matchingWordRarity'])

    # Group pairs by matchingWordRarity
    groups = {}
    for pair in sorted_pairs:
        rarity = pair['matchingWordRarity']
        if rarity in groups:
            if (pair["matchingWordRarity"] == float('inf')):
                pair["matchingWordRarity"] = -1
            groups[rarity].append(pair)
        else:
            groups[rarity] = [pair]

    # Sort pairs within each group by similarity in descending order
    for rarity, group in groups.items():
        groups[rarity] = sorted(group, key=lambda x: x['similarity'], reverse=True)

    # Flatten the list of groups and return the result
    return [pair for rarity in groups for pair in groups[rarity]]

    
# ------- PROCESS .TXT FILES ------- # 

# Read the .txt files
input_1 = open(input_path_1, 'r').read()
input_2 = open(input_path_2, 'r').read()

# Get the pairs by similarity
with open(unsorted_similarities_path, 'r') as f:
    similarity_pairs = json.load(f)

# Construct frequency table of tokens
num_tokens_1, freq_table_1 = make_token_freq_table(input_1)
num_tokens_2, freq_table_2 = make_token_freq_table(input_2)

# Split into sentences
sentences_1 = split_into_sentences(input_1)
sentences_2 = split_into_sentences(input_2)

# Tokenize sentences using tiktoken
tokenized_sentences_1 = tokenize_sentences(sentences_1)
tokenized_sentences_2 = tokenize_sentences(sentences_2)

pairs = []

# Give each sentence pair a score based on the rarity of their matching words
index = 0
for i, alpha in enumerate(tokenized_sentences_1):
    for j, beta in enumerate(tokenized_sentences_2):
        matching_score = float('inf')
        for token_a in alpha:
            for token_b in beta:
                if (token_a == token_b):
                    appearances_1 = freq_table_1[token_a] if (freq_table_1[token_a] != 0) else 1
                    appearances_2 = freq_table_2[token_b] if (freq_table_2[token_b] != 0) else 1
                    if (appearances_1 + appearances_2) < matching_score:
                        matching_score = (appearances_1 + appearances_2)
        pair = {
            1: sentences_1[i],
            2: sentences_2[j],
            "similarity": similarity_pairs[index]["similarity"],
            "matchingWordRarity": matching_score
        }
        pairs.append(pair)
        index += 1

sorted_pairs = sort_by_matching_word_rarity_then_similarity(pairs)

# ------- WRITE OUTPUT AS JSON FILE ------- # 
with open(output_path, 'w') as output:
    json.dump(sorted_pairs, output, indent = 4)

print("\n...finished writing output JSON files\n")