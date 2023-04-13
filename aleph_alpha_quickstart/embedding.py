"""This example is taken from https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/04_semantic_search.ipynb"""

import math
import os
from typing import Sequence

import numpy as np
from aleph_alpha_client import Client, Prompt, SemanticEmbeddingRequest, SemanticRepresentation
from dotenv import load_dotenv

load_dotenv()


# embedding functions from this repository https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/04_semantic_search.ipynb
def embed(text: str, representation: SemanticRepresentation):
    """helper function to embed text using the symmetric or asymmetric model.    """
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text(text), representation=representation
    )
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
        print(
            f"Similarity between '{query}' and '{text[:25]}...': {cosine_similarity(query_embedding, text_embeddings[i])}"
        )



if __name__ == "__main__":

    client = Client(token=os.getenv("AA_TOKEN"))

    query = "Who developed the first functional networks?"
    large_text = """Artificial neural networks (ANNs), usually simply called neural networks (NNs) or neural nets,[1] are computing systems inspired by the biological neural networks that constitute animal brains.[2]  
    An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives signals then processes them and can signal neurons connected to it. The "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. 
    The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold.  Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.
    Training Neural networks learn (or are trained) by processing examples, each of which contains a known "input" and "result," forming probability-weighted associations between the two, which are stored within the data structure of the net itself. The training of a neural network from a given example is usually conducted by determining the difference between the processed output of the network (often a prediction) and a target output. This difference is the error. The network then adjusts its weighted associations according to a learning rule and using this error value. Successive adjustments will cause the neural network to produce output which is increasingly similar to the target output. After a sufficient number of these adjustments the training can be terminated based upon certain criteria. This is known as supervised learning.  
    Such systems "learn" to perform tasks by considering examples, generally without being programmed with task-specific rules. For example, in image recognition, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge of cats, for example, that they have fur, tails, whiskers, and cat-like faces. Instead, they automatically generate identifying characteristics from the examples that they process.  
    History of artificial neural networks Warren McCulloch and Walter Pitts[3] (1943) opened the subject by creating a computational model for neural networks.[4] In the late 1940s, D. O. Hebb[5] created a learning hypothesis based on the mechanism of neural plasticity that became known as Hebbian learning. Farley and Wesley A. Clark[6] (1954) first used computational machines, then called "calculators", to simulate a Hebbian network. 
    In 1958, psychologist Frank Rosenblatt invented the perceptron, the first artificial neural network,[7][8][9][10] funded by the United States Office of Naval Research.[11] The first functional networks with many layers were published by Ivakhnenko and Lapa in 1965, as the Group Method of Data Handling.[12][13][14] The basics of continuous backpropagation[12][15][16][17] were derived in the context of control theory by Kelley[18] in 1960 and by Bryson in 1961,[19] using principles of dynamic programming. Thereafter research stagnated following Minsky and Papert (1969),[20] who discovered that basic perceptrons were incapable of processing the exclusive-or circuit and that computers lacked sufficient power to process useful neural networks.  
    In 1970, Seppo Linnainmaa published the general method for automatic differentiation (AD) of discrete connected networks of nested differentiable functions.[21][22] In 1973, Dreyfus used backpropagation to adapt parameters of controllers in proportion to error gradients.[23] Werbos\'s (1975) backpropagation algorithm enabled practical training of multi-layer networks. In 1982, he applied Linnainmaa\'s AD method to neural networks in the way that became widely used.[15][24]  The development of metal–oxide–semiconductor (MOS) very-large-scale integration (VLSI), in the form of complementary MOS (CMOS) technology, enabled increasing MOS transistor counts in digital electronics. This provided more processing power for the development of practical artificial neural networks in the 1980s.[25]  
    In 1986 Rumelhart, Hinton and Williams showed that backpropagation learned interesting internal representations of words as feature vectors when trained to predict the next word in a sequence.[26]  From 1988 onward,[27][28] the use of neural networks transformed the field of protein structure prediction, in particular when the first cascading networks were trained on profiles (matrices) produced by multiple sequence alignments.[29]  In 1992, max-pooling was introduced to help with least-shift invariance and tolerance to deformation to aid 3D object recognition.[30][31][32] Schmidhuber adopted a multi-level hierarchy of networks (1992) pre-trained one level at a time by unsupervised learning and fine-tuned by backpropagation.[33]"""

    text_chunks = large_text.split("\n")
    text_embeddings = [embed(text, SemanticRepresentation.Document) for text in text_chunks]
    query_embedding = embed(query, SemanticRepresentation.Query)
    # Search for the most similar split in large_text to the query and output its index
    top_index = np.argmax([cosine_similarity(query_embedding, embedding) for embedding in text_embeddings])

    print(f"The most similar split to the query is at index {top_index}:\n {text_chunks[top_index]}")