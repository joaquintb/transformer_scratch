# from sacrebleu.metrics import CHRF

# # Define toy data: model outputs and target sentences
# model_outputs = [
#     "The cat is on the mat.",
#     "Hello world!",
#     "Machine translation is amazing."
# ]

# target_sentences = [
#     "The cat sits on the mat.",
#     "Hello, world!",
#     "Machine translation is incredible."
# ]

# # Initialize the CHRF metric
# chrf = CHRF(word_order=2)  # Includes up to bigram character order

# # Calculate CHRF score
# # The function expects a list of hypotheses and a list of list(s) of references
# score = chrf.corpus_score(model_outputs, [target_sentences])

# # Print the overall CHRF score
# print(f"Average CHRF Score: {score.score:.4f}")

# # Print individual scores (optional)
# print("\nIndividual CHRF Scores:")
# for i, (output, target) in enumerate(zip(model_outputs, target_sentences)):
#     individual_score = chrf.sentence_score(output, [target])
#     print(f"Example {i+1}: CHRF Score = {individual_score.score:.4f}")

# For most translation or text generation tasks:

#     CHRF > 60: Good quality.
#     CHRF > 70: Very good quality.
#     CHRF > 80: Excellent quality (close to human-level).

# CHRF measures character n-gram overlap (precision and recall) and balances them using an F-score.
# It is robust to minor word differences, punctuation issues, and morphological variations.

import evaluate

# Load the METEOR metric from the 'evaluate' library
meteor = evaluate.load('meteor')

# Define your lists of predictions and references
predictions = [
    "This is a test",
    "Hey there"
]

references = [
    "It is a test",
    "Hello"
]

# Compute the METEOR score
results = meteor.compute(predictions=predictions, references=references)

# Print the METEOR score rounded to 2 decimal places
print(f"METEOR Score: {round(results['meteor'], 2)}")
