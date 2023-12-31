# Import required libraries
from transformers import BertModel, AutoTokenizer
import pandas as pd

# Specify the pre-trained model to use: BERT-base-cased
model_name = "bert-base-cased"

# Instantiate the model and tokenizer for the specified pre-trained model
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set a sentence for analysis
sentence = "When life gives you lemons, don't make lemonade."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)

# Create a DataFrame with the tokenizer's vocabulary
vocab = tokenizer.vocab
vocab_df = pd.DataFrame({"token": vocab.keys(), "token_id": vocab.values()})
vocab_df = vocab_df.sort_values(by="token_id").set_index("token_id")

# Encode the sentence into token_ids using the tokenizer
token_ids = tokenizer.encode(sentence)

# Print the length of tokens and token_ids
len(tokens)
len(token_ids)

# Access the tokens in the vocabulary DataFrame by index
vocab_df.iloc[101]
vocab_df.iloc[102]

# Zip tokens and token_ids (excluding the first and last token_ids for [CLS] and [SEP])
list(zip(tokens, token_ids[1:-1]))

# Decode the token_ids (excluding the first and last token_ids for [CLS] and [SEP]) back into the original sentence
tokenizer.decode(token_ids[1:-1])

# Tokenize the sentence using the tokenizer's `__call__` method
tokenizer_out = tokenizer(sentence)

# Create a new sentence by removing "don't " from the original sentence
sentence2 = sentence.replace("don't ", "")

# Tokenize both sentences with padding
tokenizer_out2 = tokenizer([sentence, sentence2], padding=True)

# Decode the tokenized input_ids for both sentences
tokenizer.decode(tokenizer_out2["input_ids"][0])
tokenizer.decode(tokenizer_out2["input_ids"][1])