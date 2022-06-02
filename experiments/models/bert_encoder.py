import logging
import pickle

import torch
import torch.nn as nn
from transformers import AutoModel, BartForConditionalGeneration, T5ForConditionalGeneration

from feedbackQA.models.general_encoder import GeneralEncoder
from feedbackQA.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


def get_ffw_layers(
        prev_hidden_size, dropout, layer_sizes, append_relu_and_dropout_after_last_layer):
    result = []
    for i, size in enumerate(layer_sizes):
        result.append(nn.Linear(prev_hidden_size, size))
        if i < len(layer_sizes) - 1 or append_relu_and_dropout_after_last_layer:
            result.append(nn.ReLU())
            result.append(nn.Dropout(p=dropout, inplace=False))
        prev_hidden_size = size
    return result


def hashable(input_id):
    return tuple(input_id.cpu().numpy())


class BertAsEncoder(GeneralEncoder):

    """
    The main BERT encode class.
    """

    def __init__(self, hyper_params, bert_model, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        if bert_model is None:
            bert = AutoModel.from_pretrained(model_hparams['bert_base'])
        else:
            bert = bert_model
        super(BertAsEncoder, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert = bert
        self.name = name

        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_probs_dropout_prob = bert_dropout
            self.bert.config.hidden_dropout_prob = bert_dropout
        else:
            logger.info('using the original bert model dropout')

        self.freeze_bert = model_hparams['freeze_bert']

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):

        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        else:
            bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        return bert_hs

class BartAsEncoder(GeneralEncoder):

    """
    The main BART encode class.
    """

    def __init__(self, hyper_params, bert_model, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        if bert_model is None:
            bert = AutoModel.from_pretrained(model_hparams['bert_base'])
        else:
            bert = bert_model
        super(BartAsEncoder, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert = bert
        self.name = name

        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_dropout = bert_dropout
            self.bert.config.dropout = bert_dropout
        else:
            logger.info('using the original bart model dropout')

        self.freeze_bert = model_hparams['freeze_bert']

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids=None):

        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return bert_hs

class DualDecBart(GeneralEncoder):

    """
    The main BART encode class.
    """

    def __init__(self, hyper_params, bert_model, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        if bert_model is None:
            bert = BartForConditionalGeneration.from_pretrained(model_hparams['bert_base'])
        else:
            bert = bert_model
        super(DualDecBart, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert_as_enc = AutoModel.from_pretrained(model_hparams['bert_base'])
        self.bert = BartForConditionalGeneration.from_pretrained(model_hparams['bert_base'])
        self.bert.model.encoder = self.bert_as_enc.encoder
        #self.bert_as_enc = BartModel(encoder=encoder, decoder=decoder_2)
        self.name = name

        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_dropout = bert_dropout
            self.bert.config.dropout = bert_dropout
        else:
            logger.info('using the original bart model dropout')

        self.freeze_bert = model_hparams['freeze_bert']

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids=None):
        assert (self.bert.model.encoder.layers[1].fc1.weight == self.bert_as_enc.encoder.layers[1].fc1.weight).all()
        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert_as_enc(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_hs, _ = self.bert_as_enc(input_ids=input_ids, attention_mask=attention_mask)
        return bert_hs

class EncoderBartEncoder(GeneralEncoder):

    """
    The main BART-encoder encode class.
    """

    def __init__(self, hyper_params, bert_model, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        if bert_model is None:
            bert = BartForConditionalGeneration.from_pretrained(model_hparams['bert_base'])
        else:
            bert = bert_model
        super(EncoderBartEncoder, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert = bert
        self.name = name

        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_dropout = bert_dropout
            self.bert.config.dropout = bert_dropout
        else:
            logger.info('using the original bart model dropout')

        self.freeze_bert = model_hparams['freeze_bert']

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids=None):

        if self.freeze_bert:
            with torch.no_grad():
                bert_hs = self.bert.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        else:
            try:
                bert_hs = self.bert.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            except:
                print(self.bert.model.encoder.__class__.__name__, 
                      self.bert.model.encoder.__class__.__name__, 
                      input_ids.size(), attention_mask.size())
        return bert_hs


class T5AsEncoder(GeneralEncoder):

    """
    The main BART encode class.
    """

    def __init__(self, hyper_params, bert_model, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        if bert_model is None:
            bert = AutoModel.from_pretrained(model_hparams['bert_base'])
        else:
            bert = bert_model
        super(T5AsEncoder, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert = bert
        self.name = name

        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_dropout = bert_dropout
            self.bert.config.dropout = bert_dropout
        else:
            logger.info('using the original bart model dropout')

        self.freeze_bert = model_hparams['freeze_bert']

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids=None):

        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return bert_hs

class T5EncAsEncoder(GeneralEncoder):

    """
    The main BART-encoder encode class.
    """

    def __init__(self, hyper_params, bert_model, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        if bert_model is None:
            bert = T5ForConditionalGeneration.from_pretrained(model_hparams['bert_base'])
        else:
            bert = bert_model
        super(T5EncAsEncoder, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert = bert
        self.name = name

        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_dropout = bert_dropout
            self.bert.config.dropout = bert_dropout
        else:
            logger.info('using the original bart model dropout')

        self.freeze_bert = model_hparams['freeze_bert']

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids=None):

        if self.freeze_bert:
            with torch.no_grad():
                bert_hs = self.bert.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        else:
            bert_hs = self.bert.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return bert_hs

class CachedBertEncoder(BertAsEncoder):

    """
    A BERT encoder with an added cache wrapper.
    Note this does not make sense if the embedding can change overtime.
    (so, use it only if the BERT model is kept frozen)
    """

    def __init__(self, hyper_params, bert_model, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert', 'cache_size'],
            model_hparams)
        super(CachedBertEncoder, self).__init__(hyper_params, bert_model, name=name)

        if not model_hparams['freeze_bert'] or not model_hparams['dropout_bert'] == 0.0:
            raise ValueError('to cache results, set freeze_bert=True and dropout_bert=0.0')
        self.cache = {}
        self.cache_hit = 0
        self.cache_miss = 0
        self.max_cache_size = model_hparams['cache_size']

    def _search_in_cache(self, input_ids, attention_mask, token_type_ids):
        results = []
        still_to_compute_iids = []
        still_to_compute_am = []
        still_to_compute_tti = []
        for i in range(input_ids.shape[0]):
            ids_hash = hashable(input_ids[i])
            if ids_hash in self.cache:
                results.append(self.cache[ids_hash].to(input_ids.device))
            else:
                results.append(None)
                still_to_compute_iids.append(input_ids[i])
                still_to_compute_am.append(attention_mask[i])
                still_to_compute_tti.append(token_type_ids[i])
        return results, still_to_compute_iids, still_to_compute_am, still_to_compute_tti

    def _store_in_cache_and_get_results(self, cache_results, bert_hs, still_to_compute_iids):
        final_results = []
        non_cached_result_index = 0
        for cache_result in cache_results:
            if cache_result is None:
                non_cached_result = bert_hs[non_cached_result_index]
                final_results.append(non_cached_result)
                if len(self.cache) < self.max_cache_size:
                    self.cache[hashable(still_to_compute_iids[non_cached_result_index])] = \
                        non_cached_result.cpu()
                non_cached_result_index += 1
            else:
                final_results.append(cache_result)
        assert non_cached_result_index == bert_hs.shape[0]
        return torch.stack(final_results, dim=0)

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):

        cache_results, still_to_compute_iids, still_to_compute_am, still_to_compute_tti = \
            self._search_in_cache(input_ids, attention_mask, token_type_ids)
        self.cache_hit += input_ids.shape[0] - len(still_to_compute_iids)
        self.cache_miss += len(still_to_compute_iids)
        if len(still_to_compute_iids) == 0:
            return torch.stack(cache_results, dim=0)

        input_ids = torch.stack(still_to_compute_iids, dim=0)
        attention_mask = torch.stack(still_to_compute_am, dim=0)
        token_type_ids = torch.stack(still_to_compute_tti, dim=0)

        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        else:
            bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)

        if self.cache is not None:
            bert_hs = self._store_in_cache_and_get_results(
                cache_results, bert_hs, still_to_compute_iids)

        return bert_hs

    def save_cache(self, save_to):
        with open(save_to, "wb") as out_stream:
            pickle.dump(self.cache, out_stream)

    def load_cache(self, load_from):
        with open(load_from, "rb") as in_stream:
            self.cache = pickle.load(in_stream)
        return len(self.cache)

    def print_stats_to(self, print_function):
        print_function('{}: cache size {} / cache hits {} / cache misses {}'.format(
            self.name, len(self.cache), self.cache_hit, self.cache_miss))
