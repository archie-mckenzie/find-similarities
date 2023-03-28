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

# Import a PDF library to make the report a PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ------- EDITABLE VARIABLES ------- #
input_path_1 = 'input/dial_of_princes.txt'
input_path_2 = 'input/wits_misery.txt'
output_path = 'output/output.json'
sorted_output_path = 'output/sorted_output.json'
report_path = 'output/report.pdf'
report_scope = 3 # how many of the most similar sentences the report should analyze
outlier_threshold = 2.7 # 2.7 is the traditional statistical threshold for outliers
explanation_model = "gpt-3.5-turbo" # use gpt-3.5-turbo or gpt-4

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

if len(similar_pairs) < report_scope: # Make sure an overlarge report_scope can't crash the program
    report_scope = len(similar_pairs) - 1

# Get an array of similarities for statistical operations
similarities = [pair["similarity"] for pair in similar_pairs]
mean = statistics.mean(similarities)
std_dev = statistics.stdev(similarities)

report = SimpleDocTemplate(report_path, pagesize=letter)

# Define styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Center', alignment=1))
styles.add(ParagraphStyle(name='Justify', alignment=4))

# Create the report content
content = []

# Write the report header
content.append(Paragraph("AUTO-GENERATED SIMILARITY REPORT", styles['Heading2']))
content.append(Spacer(1, 12))
content.append(Paragraph("Generated by: find_similarities.py", styles['BodyText']))
content.append(Spacer(1, 12))

# Write the summary statistics
content.append(Paragraph("Total sentences in first corpus: {}".format(len(embeddings_1)), styles['BodyText']))
content.append(Paragraph("Total sentences in second corpus: {}".format(len(embeddings_2)), styles['BodyText']))
content.append(Paragraph("Total pairs analyzed: {}".format(total_pairs), styles['BodyText']))
content.append(Spacer(1, 12))

content.append(Paragraph("Highest pairwise similarity: {}".format(similar_pairs[0]["similarity"]), styles['BodyText']))
content.append(Paragraph("Lowest pairwise similarity: {}".format(similar_pairs[len(similar_pairs) - 1]["similarity"]), styles['BodyText']))
content.append(Spacer(1, 12))

content.append(Paragraph("Mean pairwise similarity: {}".format(mean), styles['BodyText']))
content.append(Paragraph("Median pairwise similarity: {}".format(statistics.median(similarities)), styles['BodyText']))
content.append(Paragraph("Standard deviation: {}".format(std_dev), styles['BodyText']))
content.append(Spacer(1, 12))

# Write the most similar sentences
content.append(Paragraph("{} most similar sentences".format(report_scope), styles['Heading3']))
content.append(Spacer(1, 12))
content.append(Paragraph("The following {} sentences were most similar:".format(report_scope), styles['BodyText']))
content.append(Spacer(1, 12))
content.append(Paragraph("—", styles['BodyText']))
content.append(Spacer(1, 6))

for pair in similar_pairs[0: report_scope]:
    content.append(Paragraph('"{}"'.format(pair[1]), styles['BodyText']))
    content.append(Spacer(1, 6))
    content.append(Paragraph('"{}"'.format(pair[2]), styles['BodyText']))
    content.append(Spacer(1, 6))
    content.append(Paragraph("Similarity: {}".format(pair["similarity"]), styles['BodyText']))
    content.append(Spacer(1, 6))
    try:
        completion = openai.ChatCompletion.create(
            model = explanation_model,
            messages = [
                {"role": "system", "content": "You are a linguistic analysis assistant."},
                {"role": "user", "content": "Suggest why our vector similarity model may consider the following two sentences similar:\n\n" + pair[1] + "\n\n" + pair[2]}
            ],
            temperature = 0
        )
        content.append(Paragraph("Explanation: {}".format(completion.choices[0].message.content.strip()), styles['BodyText']))
        update_cost("You are a linguistic analysis assistant." + "Suggest why our vector similarity model may consider the following two sentences similar:\n\n" + pair[1] + "\n\n" + pair[2] + completion.choices[0].message.content, explanation_model)
    except:
        content.append(Paragraph("There has been an error and an explanation cannot be provided.\n\n"), styles['BodyText'])
    
    content.append(Spacer(1, 6))
    content.append(Paragraph("—", styles['BodyText']))
    content.append(Spacer(1, 6))
content.append(Spacer(1, 6))

content.append(Paragraph("High-similarity statistical outliers", styles['Heading3']))
content.append(Spacer(1, 12))
    # Write the high similarity statistical outliers
if (similarities[0]) < (mean + (outlier_threshold * std_dev)):
    content.append(Paragraph("There were no high-similarity statistical outliers.", styles['BodyText']))
else:
    content.append(Paragraph("The following sentence(s) were high-similarity statistical outliers:", styles['BodyText']))
    content.append(Spacer(1, 12))
    content.append(Paragraph("—", styles['BodyText']))
    content.append(Spacer(1, 6))
    for pair in similar_pairs:
        if (pair["similarity"] > (mean + (outlier_threshold * std_dev))):
            content.append(Paragraph('"{}"'.format(pair[1]), styles['BodyText']))
            content.append(Spacer(1, 6))
            content.append(Paragraph('"{}"'.format(pair[2]), styles['BodyText']))
            content.append(Spacer(1, 6))
            content.append(Paragraph("Similarity: {}".format(pair["similarity"]), styles['BodyText']))
            content.append(Spacer(1, 6))
            content.append(Paragraph("—", styles['BodyText']))
            content.append(Spacer(1, 6))
    content.append(Spacer(1, 6))

# End the report

content.append(Paragraph("End of report", styles['Heading3']))
content.append(Spacer(1, 12))
content.append(Paragraph("Total cost to produce: ${}".format(total_cost), styles['BodyText']))

# Generate the PDF report
report.build(content)