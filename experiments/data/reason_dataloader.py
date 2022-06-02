import json
import logging
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
logger = logging.getLogger(__name__)

def encode_sentence(sentence, max_length, tokenizer, **unused_args):
    input_question = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                           truncation=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
    encoding = {'ids': input_question['input_ids'].squeeze(0), 'am': input_question['attention_mask'].squeeze(0)}
    if 'token_type_ids' in input_question.keys():
        encoding['tt'] = input_question['token_type_ids'].squeeze(0)
    return encoding

def encode_sentence_pair(question, candidate, max_length, tokenizer, **unused_args):
    input_pair = tokenizer.encode_plus(question, text_pair=candidate, add_special_tokens=True,
                                           truncation=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')

    encoding = {'ids': input_pair['input_ids'].squeeze(0), 'am': input_pair['attention_mask'].squeeze(0)}
    if 'token_type_ids' in input_pair.keys():
        encoding['tt'] = input_pair['token_type_ids'].squeeze(0)
    return encoding

def get_most_agree_feedback(ratings, feedbacks):
    label_count = np.zeros(4)
    for i, rate in enumerate(ratings):
        label_count[rate] += 1.
    max_count = label_count.max()
    select_feedback, select_rating = [], []

    for i, rate in enumerate(ratings):
        if label_count[rate] == max_count:
            select_feedback.append(feedbacks[i])
            select_rating.append(rate)

    return select_feedback, select_rating

def get_max_length_feedback(feedbacks):
    lens = []
    for fb in feedbacks:
        lens.append(len(fb.split()))

    return feedbacks[np.argmax(lens)]

class ReasonDataset(Dataset):

    def __init__(self, json_file, max_example_len, max_passage_len, task_id, tokenizer, feedback_len=42):
        self.max_example_len = max_example_len
        self.max_passage_len = max_passage_len
        self.feedback_len = feedback_len
        self.tokenizer = tokenizer
        self._task_id = task_id
        self.rating_map = {"Bad": 0, "Could be Improved": 1, "Acceptable": 2, "Excellent": 3} # mean 6 5 4 3 2 1 0

        if not os.path.exists(json_file):
            raise Exception('{} not found'.format(json_file))

        with open(json_file, 'r', encoding='utf-8') as in_stream:
            json_data = json.load(in_stream)

        questions, candidates, feedbacks, ratings, mean_ratings, ratings_binary, rating_in_text = [], [], [], [], [], [], []
        rating_score = []
        reason_mean_ratings = []
        for dict_item in json_data:
            dist_rating = np.zeros(4)
            #reason_dist_rating = np.zeros(4)
            score = []
            for rating in dict_item['rating']:
                dist_rating[self.rating_map[rating]] += 1.
                score.append(self.rating_map[rating])
            dist_rating = dist_rating / dist_rating.sum()
            score = np.sum(score) / len(score)

            if 'dist_rating' in dict_item.keys():
                for rdr in dict_item['dist_rating']:
                    reason_mean_ratings.append(np.array(rdr))
            else:
                for rating in dict_item['rating']:
                    rdr = np.zeros(4)
                    rdr[self.rating_map[rating]] = 1.
                    reason_mean_ratings.append(rdr)

            
            for feedback, rating in zip(dict_item['feedback'], dict_item['rating']):

                questions.append(dict_item['question'])
                passage_text = ''
                if dict_item['passage']['reference']['page_title']:
                    passage_text += dict_item['passage']['reference']['page_title'] + '\n'
                if dict_item['passage']['reference']['section_headers']:
                    passage_text += '\n'.join(dict_item['passage']['reference']['section_headers']) + '\n'
                if dict_item['passage']['reference']['section_content']:
                    passage_text += dict_item['passage']['reference']['section_content']
                candidates.append(passage_text)
                feedbacks.append(feedback)
                ratings.append(self.rating_map[rating])
                mean_ratings.append(dist_rating)
                rating_score.append(score)

        logger.info('loading feedback data from file {}'.format(json_file))
        self.questions = questions
        self.candidates = candidates
        self.feedbacks = feedbacks
        self.ratings = ratings
        self.mean_ratings = mean_ratings
        self.rating_score = rating_score
        self.reason_mean_ratings = reason_mean_ratings
        #print('the avg feedback length is \t', np.mean([len(x.split()) for x in feedbacks]))

    def get_task_id(self):
        return self._task_id

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        candidate = self.candidates[idx]
        feedback = self.feedbacks[idx]
        rating = self.ratings[idx]
        mrate = self.mean_ratings[idx]
        reason_mrate = self.reason_mean_ratings[idx]

        bos = self.tokenizer.bos_token
        eos = self.tokenizer.eos_token
        pad = self.tokenizer.pad_token
        encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)
        encoded_candidate = encode_sentence(candidate, self.max_passage_len - self.max_example_len, self.tokenizer)
        encoded_pair = encode_sentence_pair(question, candidate, self.max_passage_len, self.tokenizer)

        decoder_input = encode_sentence(feedback, self.feedback_len, self.tokenizer)
        decoder_target = encode_sentence(feedback+pad, self.feedback_len+1, self.tokenizer)
        decoder_target = {'ids': decoder_target['ids'][1:], 'am': decoder_target['am'][1:]}
        return {
            #'question': encoded_question, 'candidate': encoded_candidate, 'pair': encoded_pair,
            'pair': encoded_pair, 'decoder_input': decoder_input, 'decoder_target': decoder_target, 'org_feedback': feedback,
            'rating': rating, 'mean_rating': mrate, 'reason_mean_rating': reason_mrate
        }
