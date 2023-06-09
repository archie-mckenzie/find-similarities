""" 


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
        completion = openai.ChatCompletion.create(
            model = explanation_model,
            messages = [
                {"role": "system", "content": "You are a linguistic analysis assistant."},
                {"role": "user", "content": "Suggest why our vector similarity model may consider the following two sentences similar:\n\n" + pair[1] + "\n\n" + pair[2]}
            ],
            temperature = 0
        )
        report.write(completion.choices[0].message.content.strip() + "\n\n")
        update_cost("You are a linguistic analysis assistant." + "Suggest why our vector similarity model may consider the following two sentences similar:\n\n" + pair[1] + "\n\n" + pair[2] + completion.choices[0].message.content, explanation_model)
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
    report.write("-----\n\n")
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

""" 