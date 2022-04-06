## 0. Introduction
This repo contains the code for our paper *Using Interactive Feedback to Improve the  Accuracy and Explainability of Question Answering Systems Post-Deployment*.


## 1. Dataset

* Download from [this repo](https://github.com/McGill-NLP/feedbackqa-data)

## 2. Library dependencies

* `pip install -r requirement.txt`
* `cd parlai && python setup.py install` 
## 3. Train RQA models

### BERT-based RQA model

* `sh scripts_qa/run_train_bert.sh`

### BART-based RQA model

* `sh scripts_qa/run_train_bart.sh`

## 4. Train Reranker models

### `Reranker1`: Train FeedbackReranker with rating feedback

* `sh scripts_rerank/run_train_rating.sh`

### `Reranker2`: Train FeedbackReranker with both rating and explanation feedback (Used in human evaluation)

* `sh scripts_rerank/run_train_feedback.sh`

### `Reranker3`: Train VanillaReranker with QA data

* `sh scripts_rerank/run_train_rating_pseudo.sh`

### `Reranker4`: Train CombinedReranker with rating feedback and QA data

* `sh scripts_rerank/run_train_rating_comb.sh`

### `Reranker5`: Train CombinedReranker with rating & explanation feedback and QA data (*Not presented in the paper*)
* Generating pseudo explanation feedback for QA data

* `sh scripts_rerank/run_gen_explain_data.sh`
* `sh scripts_rerank/run_train_feedback_comb.sh`

## 5. Inference

### QA model only
* `sh scripts_qa/run_inference.sh`

### QA model + `Reranker1` (rating feedback)
* `sh scripts_rerank/run_rerank_rate.sh`

### QA model + `Reranker2` (rating & explanation feedback)
* `sh scripts_rerank/run_rerank_feedback.sh`

### QA model + `Reranker3` (QA data)
* `sh scripts_rerank/run_rerank_rate_pseudo.sh`

### QA model + `Reranker4` (rating feedback and QA data)
* `sh scripts_rerank/run_rerank_rate_comb.sh`

### QA model + `Reranker5` (rating & explanation feedback and QA data, *Not presented in the paper*)
* `sh scripts_rerank/run_rerank_feedback_comb.sh`

## 6. Significant test for model comparisons
* Following [this paper](https://aclanthology.org/D12-1091.pdf) to compare models head-to-head
* `calc_boot_sigtest.ipynb`

## 7. Acknowledgement

The implementation of base QA models are borrowed from [ParlAI](https://github.com/facebookresearch/ParlAI). We use [Huggingface](https://github.com/huggingface) for implementing neural models.


