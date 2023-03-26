# synthesize.py
# Python script for synthesizing the output of find_similarities.py and find_matching.py
# © 2023 Archie McKenzie, MIT License

# ------- IMPORTS ------- #

# Import json for exporting output
import json

# ------- EDITABLE VARIABLES ------- #
input_json_1 = 'output/sorted_output.json'
input_json_2 = 'output/matching.json'
output_path = 'output/synthesized.json'
semantic_similarity_weight = 1
matching_word_rarity_weight = 1

# ------- PROCESS .JSON FILES ------- # 

with open(input_json_1, 'r') as f:
    semantic_similarity_pairs = json.load(f)

with open(input_json_2, 'r') as f:
    matching_word_rarity_pairs = json.load(f)

# Sort each pair in each array by Setence 1, then Setence 2
# Take newly created arrays and mash them together, adding new attributes ultimateScore and weights
# Output synthesized.json

