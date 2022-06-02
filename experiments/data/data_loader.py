import json
import logging
import os
from collections import defaultdict
import random
import math
from operator import itemgetter

from torch.utils.data import DataLoader, Dataset, ConcatDataset, DistributedSampler
from torch import stack
from torch.utils.data.sampler import Sampler, RandomSampler

logger = logging.getLogger(__name__)

OOD_STRING = '__out-of-distribution__'


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


def get_passages_by_source(json_data, keep_ood=True):
    source2passages = defaultdict(list)
    pid2passage = {}
    pid2index = {}
    for passage in json_data['passages']:

        if not is_in_distribution(passage) and not keep_ood:
            continue  # not keeping OOD

        source = passage['source']
        passage_id = passage['passage_id']
        source2passages[source].append(passage)

        if passage_id in pid2passage:
            raise ValueError('duplicate passage id: {}'.format(passage_id))

        if is_in_distribution(passage):
            pid2index[passage_id] = len(source2passages[source]) - 1
        else:
            pid2index[passage_id] = -1

        pid2passage[passage_id] = passage

    return source2passages, pid2passage, pid2index


def is_in_distribution(passage):
    ### Add out of distribution support later 
    return 1
    reference_type = passage['reference_type']
    return reference_type.lower().startswith('faq')


def _encode_passages(source2passages, max_passage_length, tokenizer, do_not_encode=False):
    """
    note - this will only encode in-distribution passages.
    :param source2passages:
    :param max_passage_length:
    :param tokenizer:
    :return:
    """
    source2encoded_passages = defaultdict(list)
    source2plaintext_passages = defaultdict(list)
    source2id = defaultdict(int)
    source2ood = defaultdict(int)
    for source, passages in source2passages.items():
        for passage in passages:
            if is_in_distribution(passage):
                passage_text = get_passage_last_header(passage) # Currently using last header
                if do_not_encode:
                    source2encoded_passages[source].append(passage_text)
                    source2plaintext_passages[source].append(passage_text)
                else:
                    encoded_passage = encode_sentence(passage_text, max_passage_length, tokenizer)
                    source2encoded_passages[source].append(encoded_passage)
                    source2plaintext_passages[source].append(passage_text)
                source2id[source] += 1
            else:
                source2ood[source] += 1

    if not do_not_encode:
        for source in source2encoded_passages.keys():
            ids, am, tt = [], [], []
            for item in source2encoded_passages[source]:
                ids.append(item['ids'])
                am.append(item['am'])
                if 'tt' in item.keys():
                    tt.append(item['tt'])
            ids = stack(ids)
            am = stack(am)
            if tt != []:
                tt = stack(tt)
                source2encoded_passages[source] = {'ids': ids, 'am': am, 'tt': tt}
            else:
                source2encoded_passages[source] = {'ids': ids, 'am': am}

    return source2encoded_passages, source2plaintext_passages, source2id, source2ood


def get_passage_content2pid(passages, duplicates_are_ok=False):
    result = defaultdict(dict)
    for passage in passages:
        source_dict = result[passage['source']]
        passage_last_header = get_passage_last_header(passage)
        if passage_last_header in source_dict and not duplicates_are_ok:
            raise ValueError('duplicate passage last header for source {}: "{}"'.format(
                passage['source'], passage_last_header
            ))
        source_dict[passage_last_header] = get_passage_id(passage)
    return result


def get_passage_last_header(passage, return_error_for_ood=False):
    if is_in_distribution(passage):
        passage_text = ''
        if passage['reference']['page_title']:
            passage_text += passage['reference']['page_title'] + '\n'
        if passage['reference']['section_headers']:
            passage_text += '\n'.join(passage['reference']['section_headers']) + '\n'
        if passage['reference']['section_content']:
            passage_text += passage['reference']['section_content']
        return passage_text
    elif return_error_for_ood:
        raise ValueError('passage is ood')
    else:
        return OOD_STRING


def get_question(example):
    return example['question']


def get_passage_id(example):
    return example['passage_id']


def get_examples(json_data, keep_ood):
    examples = []
    # always keep ood here, because we need it for the ood check later on
    _, pid2passage, _ = get_passages_by_source(json_data, keep_ood=True)
    for example in json_data['examples']:
        related_passage_id = get_passage_id(example)
        related_passage = pid2passage[related_passage_id]
        if is_in_distribution(related_passage) or keep_ood:
            examples.append(example)
    return examples


class ReRankerDataset(Dataset):

    def __init__(self, json_file, max_example_len, max_passage_len, task_id, tokenizer, keep_ood):
        self.max_example_len = max_example_len
        self.max_passage_len = max_passage_len
        self.tokenizer = tokenizer
        self._task_id = task_id

        if not os.path.exists(json_file):
            raise Exception('{} not found'.format(json_file))

        with open(json_file, 'r', encoding='utf-8') as in_stream:
            json_data = json.load(in_stream)

        source2passages, pid2passage, pid2index = get_passages_by_source(
            json_data, keep_ood=keep_ood)

        self.encoded_source2passages, self.plaintext_source2passages, source2id, source2ood = _encode_passages(
            source2passages, max_passage_len, tokenizer)
        self.pid2passage = pid2passage
        self.pid2index = pid2index
        self.examples = get_examples(json_data, keep_ood=keep_ood)
        logger.info('loaded passages from file {} - found {} sources'.format(
            json_file, len(source2id)))
        for source in source2id.keys():
            logger.info('source "{}": found {} in-distribution and {} out-of-distribution'.format(
                source, source2id[source], source2ood[source]))
        logger.info('keeping OOD? {}'.format(keep_ood))
        self.keep_ood = keep_ood

    def get_task_id(self):
        return self._task_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        question = get_question(example)
        passage_id = example['passage_id']  # this is the related passage
        encoded_question = encode_sentence(question, self.max_example_len, self.tokenizer)

        passage = self.pid2passage[passage_id]
        # this is the index of the target in the list of passages for the current source
        if is_in_distribution(passage):
            target_idx = self.pid2index[passage_id]
        else:
            if self.keep_ood:
                target_idx = -1
            else:
                raise ValueError('found ood - but keep_ood has been used..')
        source = self.pid2passage[passage_id]['source']
        label = {'ids': self.encoded_source2passages[source]['ids'][target_idx], 'am': self.encoded_source2passages[source]['am'][target_idx]}
        if 'tt' in self.encoded_source2passages[source].keys():
            label['tt'] = self.encoded_source2passages[source]['tt'][target_idx]
        label_text = self.plaintext_source2passages[source][target_idx]
        return {'question': encoded_question, 'target_idx': target_idx,
                'label': label, 'label_text': label_text,
                'passages': self.encoded_source2passages[source], 'passages_text': self.plaintext_source2passages[source]
        }


def generate_dataloader(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle, keep_ood):
    task_id = 0 ## Not Required for normal training
    dataset = ReRankerDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer, keep_ood)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def generate_dataloader_multi_files(data_files, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        n_gpus, num_workers, shuffle, keep_ood, sampling_ratios=None):
    task_id = 0 ## Not Required for normal training
    datasets = []
    for data_file in data_files:
        datasets.append(ReRankerDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer, keep_ood))
    concat_dataset = ConcatDataset(datasets)
    sampler = BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size, n_gpus=n_gpus, sampling_ratios=sampling_ratios)
    return DataLoader(concat_dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)

class BatchSchedulerSampler(Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, n_gpus=1, sampling_ratios=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_gpus = n_gpus
        self.sampling_ratios = sampling_ratios if sampling_ratios != None else [len(cur_dataset.examples) for cur_dataset in dataset.datasets]  
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.examples) for cur_dataset in dataset.datasets])
        
    def __len__(self):
        return (self.batch_size * self.n_gpus) * math.ceil(self.largest_dataset_size / (self.batch_size * self.n_gpus)) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        multiplicative_factor = math.ceil(self.largest_dataset_size/(self.batch_size * self.n_gpus)) * (len(self.dataset.datasets)/sum(self.sampling_ratios))
        sampling_numbers = [multiplicative_factor * ratio for ratio in self.sampling_ratios]
        sampling_array = []
        for i, elem in enumerate(sampling_numbers):
            sampling_array += [i] * math.ceil(elem)
        random.shuffle(sampling_array)
        # sampling_new_array = self.n_gpus * sampling_array
        sampling_new_array = []
        for elem in sampling_array:
            sampling_new_array = sampling_new_array + (self.n_gpus * [elem])
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for i in sampling_new_array:
            cur_batch_sampler = sampler_iterators[i]
            cur_samples = []
            for _ in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[i]
                    cur_samples.append(cur_sample)
                except StopIteration:
                    # got to the end of iterator - restart the iterator and continue to get samples
                    # until reaching "epoch_samples"
                    sampler_iterators[i] = samplers_list[i].__iter__()
                    cur_batch_sampler = sampler_iterators[i]
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[i]
                    cur_samples.append(cur_sample)
            final_samples_list.extend(cur_samples)
        return iter(final_samples_list)

class DatasetFromSampler(Dataset):
    """Dataset of indexes from `Sampler`."""

    def __init__(self, sampler):
        """
        Args:
            sampler (Sampler): @TODO: Docs. Contribution is welcome
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas = None,
        rank = None,
        shuffle = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))