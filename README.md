# Find the most similar sections in two large texts

## How it works

<ol>
<li>Uses NLP to split the text corpus into sentence-size chunks.</li>
<li>Embeds each sentence with OpenAI's ada-002.
</li>
<li>Calculates vector similarities of each sentence pair.
</li>
<li>Returns the top similarity match. In addition, returns any 'outlier' pairs which are 2.7σ above the mean in terms of similarity.
</li>
<li>Outputs a .txt file with every sentence pair in order of similarity.
</li>
<li>Also outputs a report written by GPT-4, which explain the similarity between the most similar sections.
</li>
</ol>

## Instructions

## Running time


