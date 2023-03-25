# Find the most similar sections in two large texts

## How it works

<ol>
<li>Uses OpenAI's tiktoken to split the text corpus into paragraph-size chunks.</li>
<li>Embeds each chunk with OpenAI's ada-002.
</li>
<li>Calculates vector similarities of each chunk pair.
</li>
<li>Returns the top similarity match. In addition, returns any 'outlier' pairs which are 2.7σ above the mean in terms of similarity.
</li>
<li>Uses GPT-4 to explain the similarity between these sections.
</li>
</ol>



