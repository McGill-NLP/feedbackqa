import logging

import torch

logger = logging.getLogger(__name__)


def get_optimizer(optimizer, model, rating_linear_layer=None):

    optimizer_name = optimizer['name']

    if optimizer_name == 'adamw':
        lr = optimizer['lr']
        logger.info('using adamw with lr={}'.format(lr))
        if rating_linear_layer:
            return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + [p for p in rating_linear_layer.parameters() if p.requires_grad], lr=lr)
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    elif optimizer_name == 'adamw_diff_lr':
        lsr = get_learning_rates(model, optimizer)
        return torch.optim.AdamW(lsr)
    elif optimizer_name == 'adam':
        lr = optimizer['lr']
        logger.info('using adam with lr={}'.format(lr))
        if rating_linear_layer:
            return torch.optim.Adam([p for p in model.parameters() if p.requires_grad] + [p for p in rating_linear_layer.parameters() if p.requires_grad], lr=lr)
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    elif optimizer_name == 'adam_diff_lr':
        lsr = get_learning_rates(model, optimizer)
        return torch.optim.Adam(lsr)
    else:
        raise ValueError('optimizer {} not supported'.format(optimizer_name))


def get_learning_rates(model, optimizer):
    ffw_lr = optimizer['ffw_lr']
    bert_lrs = optimizer['bert_lrs']
    logger.info('ffw lr={} / bert layer lrs={}'.format(ffw_lr, bert_lrs))
    lsr = [
        {'params': _get_grad_params(model.bert_question_encoder.pre_pooling_net.parameters()),
         'lr': ffw_lr},
        {'params': _get_grad_params(model.bert_paragraph_encoder.pre_pooling_net.parameters()),
         'lr': ffw_lr},
        {'params': _get_grad_params(model.bert_question_encoder.post_pooling_net.parameters()),
         'lr': ffw_lr},
        {'params': _get_grad_params(model.bert_paragraph_encoder.post_pooling_net.parameters()),
         'lr': ffw_lr}
    ]
    for i in range(12):
        layer_lr = bert_lrs[i]
        lsr.append(
            {'params': _get_grad_params(
                model.bert_question_encoder.bert.encoder.layer[i].parameters()),
             'lr': layer_lr})
        lsr.append(
            {'params': _get_grad_params(
                model.bert_paragraph_encoder.bert.encoder.layer[i].parameters()),
             'lr': layer_lr})
    return lsr


def _get_grad_params(model_params):
    return [p for p in model_params if p.requires_grad]
