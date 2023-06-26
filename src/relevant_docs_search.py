import os
import click
import glob
import pandas as pd

from rank_bm25 import BM25Okapi as BM25
import gensim
from gensim import corpora
import gensim.downloader as api
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from src.documents_retriever import get_documents

class Retriever(object):
    def __init__(self, documents: list[list[str]]):
        self.corpus = documents
        self.bm25 = BM25(self.corpus)

    def query(self, tokenized_query, n=100):
        scores = self.bm25.get_scores(tokenized_query)
        best_docs = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return best_docs, [scores[i] for i in best_docs]


class Ranker(object):
    def __init__(self, embedding_map):
        self.embedding_map = embedding_map

    def _create_mean_embedding(self, word_embeddings):
        return np.mean(
            word_embeddings,
            axis=0,
        )

    def _create_max_embedding(self, word_embeddings):
        return np.amax(
            word_embeddings,
            axis=0,
        )

    def _embed(self, tokens, embedding):
        word_embeddings = np.array([embedding[token] for token in tokens if token in embedding])
        mean_embedding = self._create_mean_embedding(word_embeddings)
        max_embedding = self._create_max_embedding(word_embeddings)
        embedding = np.concatenate([mean_embedding, max_embedding])
        unit_embedding = embedding / np.sqrt((embedding**2).sum())
        return unit_embedding

    def rank(self, tokenized_query, tokenized_documents):
        """
        Re-ranks a set of documents according to embedding distance
        """
        query_embedding = self._embed(tokenized_query, self.embedding_map) # (E,)
        document_embeddings = np.array([self._embed(document, self.embedding_map) for document in tokenized_documents]) # (N, E)
        scores = document_embeddings.dot(query_embedding)
        index_rankings = np.argsort(scores)[::-1]
        return index_rankings, np.sort(scores)[::-1]


def tokenize(document):
    return list(gensim.utils.tokenize(document.lower()))


def show_scores(documents, scores, n=10):
    for i, score in enumerate(scores):
        print("======== RANK: {} | SCORE: {} =======".format(i, score))
        print(documents[i])
        print("")
    print("\n")


def get_most_relevant(query, titles, abstracts, n_docs=10) -> list[int]:
    print('Query: "{}"'.format(query))

    corpus = [list(gensim.utils.tokenize(doc.lower())) for doc in abstracts]
    tokenized_query = tokenize(query)

    retriever = Retriever(corpus)
    retrieval_indexes, _ = retriever.query(tokenized_query, n=n_docs)

    retrieved_abstracts = [abstracts[idx] for idx in retrieval_indexes]
    retrieved_titles = [titles[idx] for idx in retrieval_indexes]

    tokenzed_retrieved_abstracts = [corpus[idx] for idx in retrieval_indexes]

    embedding_map = api.load('glove-wiki-gigaword-50')
    # embedding_map = api.load('fasttext-wiki-news-subwords-300')

    ranker = Ranker(embedding_map=embedding_map)
    ranker_indexes, ranker_scores = ranker.rank(tokenized_query, tokenzed_retrieved_abstracts)
    reranked_abstracts = [retrieved_abstracts[idx] for idx in ranker_indexes]
    reranked_titles = [retrieved_titles[idx] for idx in ranker_indexes]

    return reranked_abstracts, reranked_titles, ranker_scores

@click.command()
@click.option("--query", prompt="Search query", help="Search query")
def main(query):
    documents = get_documents()
    titles, abstracts = tuple(zip(*documents))

    reranked_abstracts, reranked_titles, ranker_scores = get_most_relevant(query, titles, abstracts)

    # show_scores(reranked_abstracts, ranker_scores)
    return reranked_abstracts, reranked_titles, ranker_scores


if __name__ == "__main__":
    main()