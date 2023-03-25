# find_similarities.py
# Python script for finding similar sentences across two large texts
# © 2023 Archie McKenzie, MIT License

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

# Import statistics for generating the report
import statistics

# ------- EDITABLE VARIABLES ------- #
input_path_1 = 'input/1.txt'
input_path_2 = 'input/2.txt'
output_path = 'output/output.json'
sorted_output_path = 'output/sorted_output.json'
report_path = 'output/report.txt'
report_scope = 3 # how many of the most similar paragraphs the report should analyze
outlier_threshold = 2.7 # 2.7 is the traditional statistical threshold for outliers

# ------- CALCULATING COSTS ------- #

total_cost = 0 # Total amount spent so far for OpenAI's models
latest_announced_cost = 0 # Last total cost printed
def update_cost(text, model):
    global total_cost, latest_announced_cost

    if model == "text-embedding-ada-002": 
        total_cost = total_cost + (0.0004 * (1 / 1000) * len(enc.encode(text))) # OpenAI quotes about ~3,000 pages per US dollar
    elif model == "gpt-3.5-turbo": 
        total_cost = total_cost + (0.002 * (1 / 1000) * len(enc.encode(text)))
    elif model == "gpt-4": 
        total_cost = total_cost + (0.06 * (1 / 1000) * len(enc.encode(text))) # roughly
    
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
    print("----- END OF SENTENCE EMBEDDING -----")
    return embeddings

# sort a list of pair jsons by the similarity index
def sort_by_similarity(pairs):
    return sorted(pairs, key=lambda x: x['similarity'], reverse=True)


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
print("----- START OF SENTENCE PAIR ANALYSIS -----")
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
        print("Analyzed sentence pair " + str(index) + "/" + str(total_pairs)) # remove this line for better performance
        # print("Similarity: " + str(cosine_similarity))
        
        pair = {
            1: sentences_1[i],
            2: sentences_2[j],
            "similarity": cosine_similarity
        }

        pairs.append(pair)
print("----- END OF SENTENCE PAIR ANALYSIS -----")

# ------- WRITE OUTPUT AS JSON FILE ------- # 
with open(output_path, 'w') as output:
    json.dump(pairs, output, indent = 4)

similar_pairs = sort_by_similarity(pairs)

with open(sorted_output_path, 'w') as output:
    json.dump(similar_pairs, output, indent = 4)

print("\n...finished writing output JSON files\n")

# ------- WRITE REPORT USING GPT ------- # 

print("writing report...\n")

# Get an array of similarities for statistical operations
similarities = [pair["similarity"] for pair in similar_pairs]
mean = statistics.mean(similarities)
std_dev = statistics.stdev(similarities)

# Write the report header
report = open(report_path, 'w')

report.write("AUTO-GENERATED SIMILARITY REPORT\n\n")
report.write("Generated by: find_similarities.py\n\n")

report.write("Total sentences in first corpus: " + str(len(embeddings_1)) + "\n")
report.write("Total sentences in second corpus: " + str(len(embeddings_2)) + "\n")
report.write("Total pairs analyzed: " + str(total_pairs) + "\n\n")

report.write("Highest pairwise similarity: " + str(similar_pairs[0]["similarity"]) + "\n")
report.write("Lowest pairwise similarity: " + str(similar_pairs[len(similar_pairs) - 1]["similarity"]) + "\n\n")

report.write("Mean pairwise similarity: " + str(mean) + "\n")
report.write("Median pairwise similarity: " + str(statistics.median(similarities)) + "\n")
report.write("Standard deviation: " + str(std_dev) + "\n\n")

report.write("------------------------------\n\n")

if len(similar_pairs) < report_scope: # Make sure an overlarge report_scope can't crash the program
    report_scope = len(similar_pairs) - 1

# Write the int(report_scope) most similar sentences and use GPT to explain why the vector model might think they are similar
report.write("The following " + str(report_scope) + " sentences were the ranked the most similar:\n\n")
report.write("-----\n\n")
for pair in similar_pairs[0: report_scope]:
    
    report.write('"' + pair[1] + '"' + "\n\n")
    report.write('"' + pair[2] + '"' + "\n\n")
    report.write("Similarity: " + str(pair["similarity"]) + "\n\n")

    report.write("Explanation: ")
    try:
        model = "gpt-3.5-turbo"
        completion = openai.ChatCompletion.create(
            model = model,
            messages = [
                {"role": "system", "content": "You are a linguistic analysis assistant."},
                {"role": "user", "content": "Suggest why our vector similarity model may consider the following two sentences similar:\n\n" + pair[1] + "\n\n" + pair[2]}
            ],
            temperature = 0
        )
        report.write(completion.choices[0].message.content.strip() + "\n\n")
        update_cost("You are a linguistic analysis assistant." + "Suggest why our vector similarity model may consider the following two sentences similar:\n\n" + pair[1] + "\n\n" + pair[2] + completion.choices[0].message.content, model)
    except:
        report.write("There has been an error and an explanation cannot be provided.\n\n")

    report.write("-----\n\n")

report.write("------------------------------\n\n")

# Write out all the high similarity statistical outliers
# This is calculated by adding an int(outlier_theshold) number of standard deviations to the mean
if (similarities[0]) < (mean + (outlier_threshold * std_dev)):
    report.write("There were no high-similarity statistical outliers.\n\n")
else:
    report.write("The following sentence(s) were high-similarity statistical outliers:\n\n")
    for pair in similar_pairs:
        if (pair["similarity"] > (mean + (outlier_threshold * std_dev))):
            report.write('"' + pair[1] + '"' +"\n\n")
            report.write('"' + pair[2] + '"' + "\n\n")
            report.write("Similarity: " + str(pair["similarity"]) + "\n\n")
            report.write("-----\n\n")

report.write("------------------------------\n\n")

# End the report
report.write("END OF REPORT\n")
report.write("TOTAL COST TO PRODUCE: $" + str(total_cost))