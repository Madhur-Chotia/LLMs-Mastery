# Import required libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch

# Load the multi_news dataset from Hugging Face and take only the 'test' split for efficiency
dataset = load_dataset("multi_news", split="test")

# Convert the test dataset to a pandas DataFrame and take only 2000 random samples
df = dataset.to_pandas().sample(2000, random_state=42)

# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode each summary in the DataFrame using the sentence transformer model and store the embeddings in a list
passage_embeddings = list(model.encode(df['summary'].to_list(), show_progress_bar=True))

# Print the shape of the first passage embedding
passage_embeddings[0].shape

# Declare a query string
query = "Find me some articles about technology and artificial intelligence"

# Define a function to find relevant news articles based on a given query
def find_relevant_news(query):
    # Encode the query using the sentence transformer model
    query_embedding = model.encode(query)
    # Print the shape of the query embedding
    query_embedding.shape

    # Calculate the cosine similarity between the query embedding and the passage embeddings
    similarities = util.cos_sim(query_embedding, passage_embeddings)

    # Find the indices of the top 3 most similar passages
    top_indicies = torch.topk(similarities.flatten(), 3).indices

    # Get the top 3 relevant passages by slicing the summaries at 200 characters and adding an ellipsis
    top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indicies]

    # Return the top 3 relevant passages
    return top_relevant_passages

# Find relevant news articles for different queries
find_relevant_news("Natural disasters")
find_relevant_news("Law enforcement and police")
find_relevant_news("Politics, diplomacy and nationalism")