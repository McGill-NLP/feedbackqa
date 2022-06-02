import json
import logging
import os

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import stack
import numpy as np

logger = logging.getLogger(__name__)

def t5_encode_sentence_pair(question, candidate, max_length, tokenizer):
    input_pair = tokenizer.encode_plus('The question is: {0} The answer is: {1} </s>'.format(question, candidate),
                                        max_length=max_length, return_tensors='pt')
                                           
    #print(input_pair)
    #print('question', question, '\t', 'candidates', candidate)
    encoding = {'ids': input_pair['input_ids'].squeeze(0), 'am': input_pair['attention_mask'].squeeze(0)}
    if 'token_type_ids' in input_pair.keys():
        encoding['tt'] = input_pair['token_type_ids'].squeeze(0)
    return encoding

def encode_sentence_batch_naive(sentences, max_length, tokenizer):
    dec_inputs = []
    dec_inputs_attn_mask = []
    for sent in sentences:
        input_ = tokenizer.encode_plus(sent, add_special_tokens=False,
                                           truncation=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
        dec_inputs.append(input_['input_ids'].squeeze(0))
        dec_inputs_attn_mask.append(input_['attention_mask'].squeeze(0))

    input_ = {'ids': stack(dec_inputs,dim=0), 'am': stack(dec_inputs_attn_mask, dim=0)}
    return input_

def encode_sentence_batch(sentences, max_length, tokenizer, add_special_tokens=True):

    #decoder_input = encode_sentence(pad + feedback, self.feedback_len, self.tokenizer)
    #decoder_target = encode_sentence(feedback + eos, self.feedback_len, self.tokenizer)
    dec_inputs = []
    dec_inputs_attn_mask = []
    tgts = []
    tgts_attn_mask = []
    for sent in sentences:
        eos = tokenizer.eos_token
        pad = tokenizer.pad_token

        input_ = pad + sent
        tgt = sent + eos
        input_ = tokenizer.encode_plus(input_, add_special_tokens=add_special_tokens,
                                           truncation=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
        tgt = tokenizer.encode_plus(tgt, add_special_tokens=add_special_tokens,
                                           truncation=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
        dec_inputs.append(input_['input_ids'].squeeze(0))
        dec_inputs_attn_mask.append(input_['attention_mask'].squeeze(0))
        tgts.append(tgt['input_ids'].squeeze(0))
        tgts_attn_mask.append(tgt['attention_mask'].squeeze(0))


    dec_inp = {'ids': stack(dec_inputs,dim=0), 'am': stack(dec_inputs_attn_mask, dim=0)}
    tgts = {'ids': stack(tgts, dim=0), 'am': stack(tgts_attn_mask, dim=0)}
    #if 'token_type_ids' in input_question.keys():
    #    encoding['tt'] = input_question['token_type_ids'].squeeze(0)
    return dec_inp, tgts

def encode_sentence(sentence, max_length, tokenizer):
    input_question = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                           truncation=True,
                                           max_length=max_length,
                                           pad_to_max_length=True,
                                           return_tensors='pt')
                                           
    
    encoding = {'ids': input_question['input_ids'].squeeze(0), 'am': input_question['attention_mask'].squeeze(0)}
    if 'token_type_ids' in input_question.keys():
        encoding['tt'] = input_question['token_type_ids'].squeeze(0)
    return encoding

class GenerationDataset(Dataset):

    def __init__(self, json_file, max_example_len, max_passage_len, task_id, tokenizer, feedback_len=40):
        self.max_example_len = max_example_len
        self.max_passage_len = max_passage_len
        self.tokenizer = tokenizer
        self._task_id = task_id
        self.rating_map = {"Bad": 0, "Could be Improved": 1, "Acceptable": 2, "Excellent": 3}
        self.t5_rating_map = {"Bad": "terrible", "Could be Improved": "bad", "Acceptable": "good", "Excellent": "excellent"}
        self.feedback_len = feedback_len

        if not os.path.exists(json_file):
            raise Exception('{} not found'.format(json_file))

        with open(json_file, 'r', encoding='utf-8') as in_stream:
            json_data = json.load(in_stream)

        questions, candidates, feedbacks, ratings, rating_in_text, mean_ratings = [], [], [], [], [], []
        for dict_item in json_data:
            dist_rating = np.zeros(4)
            for rating in dict_item['rating']:
                dist_rating[self.rating_map[rating]] += 1.
            dist_rating = dist_rating / dist_rating.sum()
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
                rating_in_text.append(self.t5_rating_map[rating])
                mean_ratings.append(dist_rating)
        logger.info('loading feedback data from file {}'.format(json_file))
        self.questions = questions
        self.candidates = candidates
        self.feedbacks = feedbacks
        self.ratings = ratings
        self.mean_ratings = mean_ratings
        self.rating_in_text = rating_in_text

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

        if not 'T5' in str(self.tokenizer.__class__):
            tok = self.tokenizer.cls_token
            pad = self.tokenizer.pad_token

            encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)
            encoded_candidate = encode_sentence(candidate, self.max_passage_len - self.max_example_len, self.tokenizer)
            encoded_pair = encode_sentence(question+tok+candidate, self.max_passage_len, self.tokenizer)

            decoder_input = encode_sentence(tok + feedback, self.feedback_len, self.tokenizer)
            decoder_target = encode_sentence(feedback + pad, self.feedback_len, self.tokenizer)
        else:
            eos = self.tokenizer.eos_token
            pad = self.tokenizer.pad_token

            encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)
            encoded_candidate = encode_sentence(candidate, self.max_passage_len - self.max_example_len, self.tokenizer)

            decoder_input = encode_sentence(pad + feedback, self.feedback_len, self.tokenizer)
            decoder_target = encode_sentence(feedback + eos, self.feedback_len, self.tokenizer)

        return {
            'question': encoded_question, 'candidate': encoded_candidate, 'qa_pair': encoded_pair,
            'decoder_input': decoder_input, 'decoder_target': decoder_target, 'rating': rating, 'mean_rating': mrate
        }

class T5GenerationDataset(GenerationDataset):
    def __getitem__(self, idx):
        question = self.questions[idx]
        candidate = self.candidates[idx]
        feedback = self.feedbacks[idx]
        rating = self.ratings[idx]

        eos = self.tokenizer.eos_token
        pad = self.tokenizer.pad_token
        
        encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)
        encoded_candidate = encode_sentence(candidate, self.max_passage_len - self.max_example_len, self.tokenizer)

        encoded_qa_pair = encode_sentence('The question is: {0} The answer is: {1} </s>'.format(question, candidate),
                                            self.max_example_len, self.tokenizer)
        #good_prompt = encode_sentence('The question is: {0} The answer is: {1} This answer is good. </s>'.format(question, candidate))
        #bad_prompt = encode_sentence('The question is: {0} The answer is: {1} This answer is bad. </s>'.format(question, candidate))
        decoder_input = encode_sentence(pad + feedback, self.feedback_len, self.tokenizer)
        decoder_target = encode_sentence(feedback + eos, self.feedback_len, self.tokenizer)

        return {
            'org_question': question, 'org_candidate': candidate,
            'question': encoded_question, 'candidate': encoded_candidate, 'pair': encoded_qa_pair, 
            'decoder_input': decoder_input, 'decoder_target': decoder_target, 'rating': rating
        }

class T5MTLDataset(GenerationDataset):
    def __getitem__(self, idx):
        question = self.questions[idx]
        candidate = self.candidates[idx]
        feedback = self.feedbacks[idx]
        rating = self.ratings[idx]

        eos = self.tokenizer.eos_token
        pad = self.tokenizer.pad_token
        
        encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)
        encoded_candidate = encode_sentence(candidate, self.max_passage_len - self.max_example_len, self.tokenizer)

        encoded_qa_pair = encode_sentence('The question is: {0} The answer is: {1} The rating is <extra_id_1> </s>'.format(question, candidate),
                                            self.max_example_len, self.tokenizer)
        #good_prompt = encode_sentence('The question is: {0} The answer is: {1} This answer is good. </s>'.format(question, candidate))
        #bad_prompt = encode_sentence('The question is: {0} The answer is: {1} This answer is bad. </s>'.format(question, candidate))
        decoder_input = encode_sentence(pad + feedback, self.feedback_len, self.tokenizer)
        decoder_target = encode_sentence(feedback + eos, self.feedback_len, self.tokenizer)

        return {
            'org_question': question, 'org_candidate': candidate,
            'question': encoded_question, 'candidate': encoded_candidate, 'pair': encoded_qa_pair, 
            'decoder_input': decoder_input, 'decoder_target': decoder_target, 'rating': rating
        }

def generate_dataloader(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    task_id = 0 ## Not Required for normal training
    dataset = GenerationDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def generate_dataloader_multi_files(data_files, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    task_id = 0 ## Not Required for normal training
    datasets = []
    for data_file in data_files:
        datasets.append(GenerationDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer))
    concat_dataset = ConcatDataset(datasets)
    return DataLoader(concat_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

class GPTGenerationDataset(Dataset):

    def __init__(self, json_file, max_example_len, max_passage_len, task_id, tokenizer):
        self.max_example_len = max_example_len
        self.max_passage_len = max_passage_len
        self.tokenizer = tokenizer
        self._task_id = task_id
        self.rating_map = {"Bad": 0, "Could be Improved": 1, "Acceptable": 2, "Excellent": 3}

        if not os.path.exists(json_file):
            raise Exception('{} not found'.format(json_file))

        with open(json_file, 'r', encoding='utf-8') as in_stream:
            json_data = json.load(in_stream)

        questions, candidates, feedbacks, ratings = [], [], [], [] 
        for dict_item in json_data:
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
        logger.info('loading feedback data from file {}'.format(json_file))
        self.questions = questions
        self.candidates = candidates
        self.feedbacks = feedbacks
        self.ratings = ratings

    def get_task_id(self):
        return self._task_id

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        candidate = self.candidates[idx]
        feedback = self.feedbacks[idx]
        rating = self.ratings[idx]

        tok = self.tokenizer.cls_token
        pad = self.tokenizer.pad_token
        
        encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)
        encoded_candidate = encode_sentence(candidate, self.max_passage_len - self.max_example_len, self.tokenizer)

        decoder_input = encode_sentence(tok + feedback, self.max_passage_len, self.tokenizer)
        decoder_target = encode_sentence(feedback + pad, self.max_passage_len, self.tokenizer)

        return {
            'question': encoded_question, 'candidate': encoded_candidate,
            'decoder_input': decoder_input, 'decoder_target': decoder_target, 'rating': rating
        }


def generate_dataloader(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    task_id = 0 ## Not Required for normal training
    dataset = GenerationDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def generate_dataloader_multi_files(data_files, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    task_id = 0 ## Not Required for normal training
    datasets = []
    for data_file in data_files:
        datasets.append(GenerationDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer))
    concat_dataset = ConcatDataset(datasets)
    return DataLoader(concat_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)