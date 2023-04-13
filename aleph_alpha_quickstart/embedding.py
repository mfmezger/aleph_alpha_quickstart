"""This example is taken from https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/04_semantic_search.ipynb"""

import math
import os
from typing import Sequence

import numpy as np
from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)
from dotenv import load_dotenv

load_dotenv()


# embedding functions from this repository https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/04_semantic_search.ipynb
def embed(text: str, representation: SemanticRepresentation):
    """helper function to embed text using the symmetric or asymmetric model."""
    request = SemanticEmbeddingRequest(prompt=Prompt.from_text(text), representation=representation)
    result = client.semantic_embed(request, model="luminous-base")
    return result.embedding


# helper function to calculate the cosine similarity between two vectors
def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


# helper function to print the similarity between the query and text embeddings
def print_result(texts, query, query_embedding, text_embeddings):
    for i, text in enumerate(texts):
        print(f"Similarity between '{query}' and '{text[:25]}...': {cosine_similarity(query_embedding, text_embeddings[i])}")


if __name__ == "__main__":
    client = Client(token=os.getenv("AA_TOKEN"))

    query = "Who developed the first functional networks?"
    # load the text from the file ressources/large_text.txt
    with open("ressources/large_text.txt") as f:
        large_text = f.read()

    text_chunks = large_text.split("\n")
    text_embeddings = [embed(text, SemanticRepresentation.Document) for text in text_chunks]
    query_embedding = embed(query, SemanticRepresentation.Query)
    # Search for the most similar split in large_text to the query and output its index
    top_index = np.argmax([cosine_similarity(query_embedding, embedding) for embedding in text_embeddings])

    print(f"The most similar split to the query is at index {top_index}:\n {text_chunks[top_index]}")
