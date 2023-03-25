# find_similarities.py
# script for finding similar sections in large chunks of text

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

# ------- EDITABLE VARIABLES ------- #
input_path_1 = 'input/1.txt'
input_path_2 = 'input/2.txt'
output_path = 'output/output.txt'

# ------- SPLITTING INTO SENTENCES ------- #

# Split any given text into sentences
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# ------- PROCESS .TXT FILES ------- # 

# Read the .txt files
input_1 = open(input_path_1, 'r').read()
input_2 = open(input_path_2, 'r').read()

# Split into sentences
sentences_1 = split_into_sentences(input_1)
sentences_2 = split_into_sentences(input_2)

# ------- CREATE EMBEDDINGS ------- # 

# ------- COMPARE EACH PAIR OF EMBEDDINGS ------- # 