import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from feedbackQA.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


def compute_average_with_padding(tensor, padding):
    """

    :param tensor: dimension batch_size, seq_length, hidden_size
    :param padding: dimension batch_size, seq_length
    :return:
    """
    batch_size, seq_length, emb_size = tensor.shape
    expanded_padding = padding.unsqueeze(-1).repeat(1, 1, emb_size)
    padded_tensor = tensor * expanded_padding
    entry_sizes = torch.sum(padding, axis=1).unsqueeze(1).repeat(1, emb_size)
    return torch.sum(padded_tensor, axis=1) / entry_sizes


def _get_layers(prev_hidden_size, dropout, layer_sizes, append_relu_and_dropout_after_last_layer):
    result = []
    for i, size in enumerate(layer_sizes):
        result.append(nn.Linear(prev_hidden_size, size))
        if i < len(layer_sizes) - 1 or append_relu_and_dropout_after_last_layer:
            result.append(nn.ReLU())
            result.append(nn.Dropout(p=dropout, inplace=False))
        prev_hidden_size = size
    return result


class GeneralEncoder(nn.Module):

    def __init__(self, hyper_params, encoder_hidden_size):
        super(GeneralEncoder, self).__init__()

        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['layers_pre_pooling', 'layers_post_pooling', 'dropout',
             'normalize_model_result', 'pooling_type'],
            model_hparams)

        self.pooling_type = model_hparams['pooling_type']
        self.normalize_model_result = model_hparams['normalize_model_result']

        pre_pooling_seq = _get_layers(encoder_hidden_size, model_hparams['dropout'],
                                      model_hparams['layers_pre_pooling'],
                                      True)
        self.pre_pooling_net = nn.Sequential(*pre_pooling_seq)

        pre_pooling_last_hidden_size = model_hparams['layers_pre_pooling'][-1] if \
            model_hparams['layers_pre_pooling'] else encoder_hidden_size
        post_pooling_seq = _get_layers(pre_pooling_last_hidden_size, model_hparams['dropout'],
                                       model_hparams['layers_post_pooling'],
                                       False)
        self.post_pooling_last_hidden_size = model_hparams['layers_post_pooling'][-1] if \
            model_hparams['layers_post_pooling'] else pre_pooling_last_hidden_size
        self.post_pooling_net = nn.Sequential(*post_pooling_seq)

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):
        raise ValueError('not implemented - use a subclass')

    def forward(self, input_ids, attention_mask, token_type_ids=None, dummy_tensor=None, pooling_type=None):
        hs = self.get_encoder_hidden_states(input_ids=input_ids, attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)

        pre_pooling_hs = self.pre_pooling_net(hs)

        if pooling_type is None: 
            pooling_type = self.pooling_type
        if pooling_type == 'cls':
            result_pooling = pre_pooling_hs[:, 0, :]
        elif pooling_type == 'avg':
            result_pooling = compute_average_with_padding(pre_pooling_hs, attention_mask)
        elif pooling_type == 'polyencoder_context':
            return pre_pooling_hs, attention_mask
        elif pooling_type == 'eos_mask':
            eos_mask = input_ids.eq(self.bert.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            result_pooling = pre_pooling_hs[eos_mask, :].view(pre_pooling_hs.size(0), -1, pre_pooling_hs.size(-1))[:, -1, :]
        else:
            raise ValueError('pooling {} not supported.'.format(self.pooling_type))

        post_pooling_hs = self.post_pooling_net(result_pooling)

        if self.normalize_model_result:
            return F.normalize(post_pooling_hs)
        else:
            return post_pooling_hs
