import json
from typing import List

from feedbackQA.models.model_helper import build_retriever, build_reranker, build_tokenizer, prepare_rerank_input

import numpy as np
import torch
#from .feedback_system_abc import FeedbackSystemABC
#from .samples import SAMPLE_FEEDBACK


class FQA():
    def __init__(self, ret_hparams, rank_hparams, retriever_ckpt_path, reranker_ckpt_path,
                regions=["Australia", "CDC", "UK", "WHO"], topk=5, return_n=5):
        self.regions = regions
        self.reranker = build_reranker(rank_hparams, reranker_ckpt_path)
        self.retriever = build_retriever(ret_hparams, retriever_ckpt_path).to('cpu')
        self.topk = topk
        self.max_qa_pair_length = rank_hparams['max_paragraph_len']+rank_hparams['max_question_len']
        self.max_decode_length = 40#rank_hparams['max_decode_length']
        self.rating_map = {0: "Bad 😟", 1: "Needs improvement 😐", 2: "Acceptable 🙂", 3: "Excellent 😆"}
        self.embedded_candidates = {}
        self.return_n = return_n
        self.computing = False
        #self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)

    def load_kb_files(self, load_dir: str, verbose=False):
        """
        Parameters:
            load_dir:
                The directory where we are loading JSON files containing each groups
        Returns:
            passages:
                dictionary of region (str) mapping to passages (list of dictionaries)
        """
        passages = {}

        for region in self.regions:
            with open(f"{load_dir}/passages_{region}.json", "rb") as f:
                #passages.extend(json.load(f))
                psges = [x['content'] for x in json.load(f)]#[:10]
                passages[region] = psges
        return passages

    def build_knowledge_base(self, passages: List[dict]):
        for region in self.regions:
            self.embedded_candidates[region] = self.retriever.retriever.embed_paragrphs(passages[region])
        self.passages = passages
        #self.contents = np.array([p["content"] for p in self.passages])

    def retrieve_and_rerank(self, region, query: str):
        print('Retrieving....')
        #torch.cuda.empty_cache()
        self.reranker.to(torch.device('cpu'))
        self.retriever.to(torch.device('cuda'))
        _, pre_select, qa_score = self.retriever.retriever.predict(
                                          query,
                                          self.embedded_candidates[region],
                                          passages_already_embedded=True,
                                          question_already_embedded=False,
                                          topk=self.topk)
        pre_select = pre_select.squeeze(0)
        torch.cuda.empty_cache()
        candidates = [self.passages[region][id_] for id_ in pre_select.tolist()]
        query_k = [query] * len(pre_select)
        print('Reranking....')
        self.retriever.to(torch.device('cpu'))
        self.reranker.to(torch.device('cuda'))
        pair_input = prepare_rerank_input(query_k, candidates, self.reranker.tokenizer, self.max_qa_pair_length)
        rating_cls, fb_score, feedbacks = self.reranker.rerank_and_feedback(pair_input, self.max_decode_length)
        final_score = (fb_score + qa_score).detach().cpu().numpy()[0]
        indices = np.argsort(final_score)[::-1][:self.return_n]
        #best = np.argmax(final_score)
        best_idx = [pre_select.cpu()[x] for x in indices]
        rating_cls = [rating_cls.cpu().tolist()[x] for x in indices]
        rating_text = [self.rating_map[cls_] for cls_ in rating_cls]
        feedback = [feedbacks[x] for x in indices]
        answer = [self.passages[region][idx] for idx in best_idx]
        print(answer, rating_text, feedback)
        return answer, rating_text, feedback

    def output(self, region: str, query: str):
        answer, rating, feedback = self.retrieve_and_rerank(region, query)
        #best_candidates = self.passages[region][best_idx]#.tolist()
        #feedback = "This answer is {0}; Explaination: {1}".format(rating, feedback)
        return answer, rating, feedback