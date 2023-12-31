from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine

model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def predict(text):
    encoded_inputs = tokenizer(text, return_tensors="pt")

    return model(**encoded_inputs)[0]


sentence1 = "There was a fly drinking from my soup"
sentence2 = "There is a fly swimming in my juice"
# sentence2 = "To become a commercial pilot, he had to fly for 1500 hours."

tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)

out1 = predict(sentence1)
out2 = predict(sentence2)

emb1 = out1[0:, tokens1.index("fly"), :].detach()
emb2 = out2[0:, tokens2.index("fly"), :].detach()

# emb1 = out1[0:, 3, :].detach()
# emb2 = out2[0:, 3, :].detach()


emb1.shape
emb2.shape

cosine(emb1, emb2)
