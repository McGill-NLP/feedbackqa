import json
from typing import List

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except:
    error_msg = (
        "Couldn't import scikit-learn. To use the toy models, you will need to "
        "install it with `pip install scikit-learn`."
    )
    raise Exception(error_msg)

import numpy as np

from .feedback_system_abc import FeedbackSystemABC
from .samples import SAMPLE_FEEDBACK


class ToyFeedbackRetriever:
    """
    Toy feedback retriever
    """

    def __init__(self):
        self.svd = TruncatedSVD(300)
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)

        # build knowledge base here:
        self.sample_feedback = SAMPLE_FEEDBACK
        feedback_tfidf = self.vectorizer.fit_transform(self.sample_feedback)
        self.feedback_encs = self.svd.fit_transform(feedback_tfidf)

    def get_feedback(self, query: str) -> str:
        enc = self.vectorizer.transform([query])
        ls = self.svd.transform(enc)
        idx = cosine_similarity(ls, self.feedback_encs).argmax()
        return self.sample_feedback[idx]


class ToyFQA(FeedbackSystemABC):
    def __init__(self, regions=["Australia", "CDC", "UK", "WHO"]):
        self.feedback_retriever = ToyFeedbackRetriever()

        self.regions = regions

        self.svd = TruncatedSVD(300)
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)

    def load_kb_files(self, load_dir: str, verbose=False):
        """
        Parameters:
            load_dir:
                The directory where we are loading JSON files containing each groups
        Returns:
            passages:
                dictionary of region (str) mapping to passages (list of dictionaries)
        """
        passages = []

        for region in self.regions:
            with open(f"{load_dir}/passages_{region}.json", "rb") as f:
                passages.extend(json.load(f))

        return passages

    def build_knowledge_base(self, passages: List[dict]):
        self.passages = np.array(passages)
        self.contents = np.array([p["content"] for p in self.passages])
        content_tfidf = self.vectorizer.fit_transform(self.contents)
        self.content_encs = self.svd.fit_transform(content_tfidf)

    def retrieve_idx(self, query: str, k: int = 10) -> List[int]:
        enc = self.vectorizer.transform([query])
        ls = self.svd.transform(enc)
        sim_scores = cosine_similarity(ls, self.content_encs).squeeze()

        best_idx = sim_scores.argsort()[::-1][:k].tolist()

        return best_idx

    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        best_idx = self.retrieve_idx(query, k)
        best_candidates = self.passages[best_idx].tolist()

        return best_candidates

    def rate(self, query: str, candidates: List[str]) -> List[str]:
        rand_idx = np.random.randint(4, size=len(candidates))
        choices = ["Bad ☹️", "Needs improvement 😐", "Acceptable 🙂", "Excellent 😀"]
        return [choices[i] for i in rand_idx]

    def give_feedback(
        self, query: str, candidates: List[str], verbose: bool = True
    ) -> List[str]:
        if self.feedback_retriever is None:
            return "Unable to give feedback."

        return [self.feedback_retriever.get_feedback(query + c) for c in candidates]
