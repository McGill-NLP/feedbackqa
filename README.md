# FeedbackQA

> [**Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment**](https://arxiv.org/abs/2204.03025)
> *Zichao Li, Prakhar Sharma, Xing Han Lu, Jackie C.K. Cheung, Siva Reddy*

![An diagram of how FeedbackQA conceptually works](/docs/fbqa.png)

This repository provides the code for experiments and the dataset we released. You can find below useful links:

[📄 Read](https://arxiv.org/abs/2204.03025)\
[:octocat: Code](https://github.com/McGill-NLP/feedbackqa)\
[🔗 Webpage](https://mcgill-nlp.github.io/feedbackqa/)\
[💻 Demo](http://206.12.100.48:8080/)\
[🤗 Huggingface Dataset](https://huggingface.co/datasets/McGill-NLP/feedbackQA)


You can cite us by using the following bibtex entry:
```
@inproceedings{feedbackqa2022,
  title={Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment},
  author={Zichao Li, Prakhar Sharma, Xing Han Lu, Jackie C.K. Cheung, Siva Reddy},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  year={2022}
}
```

## Experiments

The full instructions for running the experiments described in the paper can be found in [`experiments/`](./experiments).

## Dataset

You can find link to Huggingface Dataset at the top of the readme. If you want to download them directly, you can find the data in [`data/`](./data). If you want to unprocessed, original version of the data, you can find them in [`experiments/feedbackQA_data/`](./experiments/feedbackQA_data/). Please refer to the instructions in [`experiments/`](./experiments) for more information on the structure of the original data.

### Direct download

To download directly from command line, run:

```bash
wget https://github.com/McGill-NLP/feedbackqa/raw/main/data/feedback_train.json
wget https://github.com/McGill-NLP/feedbackqa/raw/main/data/feedback_valid.json
wget https://github.com/McGill-NLP/feedbackqa/raw/main/data/feedback_test.json
```

### Cloned repository

If you clone the repository, you will be able to directly access the dataset:

```bash
git clone https://github.com/McGill-NLP/feedbackqa
cd feedbackqa/data
```

### Data usage

The dataset is stored in the JSON format since it is structured in a particular way. We recommend loading it with Python:

```python
import json

# Let's load the validation set and take a single sample as an example
valid = json.load(open('feedback_valid.json'))
sample = valid[0]

print(sample.keys())
# => dict_keys(['question', 'passage', 'feedback', 'rating', 'domain'])

print(sample['question'])
# => What are the new guidelines for DSS income reporting?

print(sample['passage'].keys())
# => dict_keys(['passage_id', 'source', 'uri', 'reference_type', 'reference'])
print(sample['passage']['reference'].keys())
# => dict_keys(['page_title', 'section_headers', 'section_content', 'selection_span', 'section_content_html'])
print(sample['passage']['reference']['section_content'])
# => If we approve your claim, you’ll need to report your\nincome for the past 2 weeks to\nget your first payment.\nTo do this, [...]

print(sample['feedback'])
# => ['Directs people to the tools where they can report their income.', 'Gives requirements and links in response.']

print(sample['rating'])
# => ['Excellent', 'Excellent']

print(sample['domain'])
# => Australia
```


