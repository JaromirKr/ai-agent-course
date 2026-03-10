from langchain.tools import tool
import math
from collections import Counter, defaultdict

def compute_tf(doc_words):
    count = Counter(doc_words)
    total = len(doc_words) or 1
    return {word: count[word] / total for word in count}

def compute_idf(docs):
    N = len(docs) or 1
    df = defaultdict(int)
    for doc in docs:
        for word in set(doc):
            df[word] += 1
    return {word: math.log(N / df[word]) for word in df}

def tokenize(text):
    return [
        w.strip(",.!?()[]{}:;\"'").lower()
        for w in text.split()
        if len(w) > 3
    ]

@tool
async def extract_keywords_tfidf(text: str, top_n: int = 10) -> list:
    """
        Extract top-N keywords from text using a simple TF-IDF algorithm.
        The text is split into paragraphs, tokenized, and scored.
        Returns a list of keywords sorted by TF-IDF weight.
    """
    print("Running extract_keywords_tfidf tool...")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    docs = [tokenize(p) for p in paragraphs]

    tfs = [compute_tf(doc) for doc in docs]
    idf = compute_idf(docs)

    tfidf = defaultdict(float)
    for tf in tfs:
        for word, tf_value in tf.items():
            tfidf[word] += tf_value * idf[word]

    sorted_words = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]
