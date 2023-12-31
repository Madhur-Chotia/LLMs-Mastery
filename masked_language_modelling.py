# Import required libraries
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.special import softmax
import numpy as np

# Specify the pre-trained model to use: BERT-base-cased
model_name = "bert-base-cased"

# Instantiate the tokenizer and model for the specified pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Get the mask token from the tokenizer
mask = tokenizer.mask_token

# Create a sentence with a mask token to be filled in by the model
sentence = f"I want to {mask} pizza for tonight."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)

# Encode the sentence using the tokenizer and return the input tensors
encoded_inputs = tokenizer(sentence, return_tensors='pt')

# Get the model's output for the input tensors
outputs = model(**encoded_inputs)
# Detach the logits from the model's output and convert them to numpy arrays
logits = outputs.logits.detach().numpy()[0]

# Extract the logits for the mask token
mask_logits = logits[tokens.index(mask) + 1]
# Calculate the confidence scores for each possible token using softmax
confidence_scores = softmax(mask_logits)

# Print the top 5 predicted tokens and their confidence scores
for i in np.argsort(confidence_scores)[::-1][:5]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]

    # Print the predicted sentence with the mask token replaced by the predicted token, and the confidence score
    print(sentence.replace(mask, pred_token), score)