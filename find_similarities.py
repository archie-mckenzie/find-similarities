# find_similarities.py
# Python script for finding similar sections in large chunks of text
# © Archie McKenzie, MIT License

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

# Import OpenAI, loading the API key from the .env file
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Import tiktoken, for calculating OpenAI model costs
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Import numpy for calculating similarity between vectors
import numpy as np

# Import json for exporting output
import json


# ------- EDITABLE VARIABLES ------- #
input_path_1 = 'input/1.txt'
input_path_2 = 'input/2.txt'
output_path = 'output/output.json'
report_path = 'output/report.txt'

# ------- CALCULATING TIMING AND COSTS ------- #

total_cost = 0 # Total amount spent so far for OpenAI's models
latest_announced_cost = 0 # Last total cost printed
def update_cost(text, model):
    global total_cost, latest_announced_cost
    if model == "text-embedding-ada-002": # OpenAI quotes about ~3,000 pages per US dollar
        total_cost = total_cost + (0.0004 * len(enc.encode(text)))
    if (total_cost > latest_announced_cost + 1):
        latest_announced_cost = total_cost
        print("$" + str(total_cost) + " spent so far")


# ------- OTHER FUNCTIONS ------- #

# Split any given string into sentences
# Return an array of strings
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Convert any given array of strings into vector embeddings
# Returns an array of embeddings (embeddings are arrays of doubles)
def embed_sentences(sentences):
    embeddings = []

    index = 0 # which sentence are we on?
    total_sentence_number = len(sentences) # how many sentences are there?

    print("----- START OF SENTENCE EMBEDDING -----")
    for sentence in sentences:
        response = openai.Embedding.create(
            input = sentence,
            model = "text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
        index += 1
        print("Embedded sentence " + str(index) + "/" + str(total_sentence_number))
        update_cost(sentence, "text-embedding-ada-002")
    
    return embeddings

# ------- PROCESS .TXT FILES ------- # 

# Read the .txt files
input_1 = open(input_path_1, 'r').read()
input_2 = open(input_path_2, 'r').read()

# Split into sentences
sentences_1 = split_into_sentences(input_1)
sentences_2 = split_into_sentences(input_2)

# ------- CREATE EMBEDDINGS ------- # 

embeddings_1 = embed_sentences(sentences_1)
embeddings_2 = embed_sentences(sentences_2)

# ------- COMPARE EACH PAIR OF EMBEDDINGS ------- # 

pairs = [] 

index = 0 # which pair are we on?
total_pairs = len(embeddings_1) * len(embeddings_2) # how many pairs are there?
for i, alpha in enumerate(embeddings_1):
    alpha = np.array(alpha)
    for j, beta in enumerate(embeddings_2):
        beta = np.array(beta)

        # Calculate the dot product of the two vectors
        dot_product = np.dot(alpha, beta)

        # Calculate the norm (length) of each vector
        alpha_norm = np.linalg.norm(alpha)
        beta_norm = np.linalg.norm(beta)

        # Calculate the cosine similarity between the two vectors
        cosine_similarity = dot_product / (alpha_norm * beta_norm)

        index += 1
        print("Analyzed sentence pair " + str(index) + "/" + str(total_pairs))
        print("Similarity: " + str(cosine_similarity))
        
        pair = {
            1: sentences_1[i],
            2: sentences_2[j],
            "similarity": cosine_similarity
        }

        pairs.append(pair)

with open(output_path, 'w') as output:
    json.dump(pairs, output, indent = 4)