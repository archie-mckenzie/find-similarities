# Find the most similar sentences in two large texts

## How it works

<ol>
<li>
Uses NLTK to split the text corpus into sentence-size chunks.
</li>
<li>
Embeds each sentence with OpenAI's ada-002.
</li>
<li>
Calculates vector similarities of each sentence pair.
</li>
<li>
Outputs a .json file, called `output.json` by default, with every sentence pair in order of writing.
</li>
<li>
Outputs a .json file, called `sorted_output.json` by default, with every sentence pair in order of similarity.
</li>
<li>
Outputs a .txt file, called `report.txt` by default. This report contains information like the mean, standard deviation, and cost it took to produce the outputs. It also includes a selection of highly similar outlier sentences, by default defined as 2.7σ above the mean similarity score.
</li>
</ol>

## Instructions

<ol>
<li>Download the repo</li>
<li>Install Python</li>
<li>Install this project's dependencies, including:
<ul>
<li>nltk</li>
<li>openai</li>
<li>dotenv</li>
<li>tiktoken</li>
<li>numpy</li>
</ul>
</li>
<li>Create a file called `.env` with `OPENAI_API_KEY = [your openai api key]`</li>
<li>Run `find_similarities.py`</li>
<li>Inspect the output folder for results</li>
</ol>

Note: the limiting time factor of `find_similarities.py` is the embedding with ada-002, which must be done for every sentence in both input texts.


