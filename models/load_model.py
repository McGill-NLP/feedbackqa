import logging
import re

from feedbackQA.models.bert_encoder import BertAsEncoder, BartAsEncoder, EncoderBartEncoder, \
                                                CachedBertEncoder, T5EncAsEncoder, DualDecBart
from feedbackQA.models.retriever import EmbeddingRetriever, FeedForwardRetriever, PolyEncoderRetriever, InversePolyEncoderRetriever
from feedbackQA.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


def load_model(hyper_params, tokenizer, debug):
    check_and_log_hp(['name', 'single_encoder'], hyper_params['model'])
    if hyper_params['model']['name'] == 'bert_encoder':
        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder_class = CachedBertEncoder
        elif re.search('bart', hyper_params['model'].get('bert_base')):
            if hyper_params['model'].get('encoder_only', None):
                encoder_class = EncoderBartEncoder
            else:
                encoder_class = BartAsEncoder
        else:
            encoder_class = BertAsEncoder

        bert_paragraph_encoder, bert_question_encoder = _create_encoders(encoder_class,
                                                                         hyper_params)

        model = EmbeddingRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], debug)
    elif hyper_params['model']['name'] == 'bert_ffw':

        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder_class = CachedBertEncoder
        elif re.search('bart', hyper_params['model'].get('bert_base')):
            if hyper_params['model'].get('encoder_only', None):
                encoder_class = EncoderBartEncoder
            else:
                encoder_class = BartAsEncoder
        else:
            encoder_class = BertAsEncoder

        bert_paragraph_encoder, bert_question_encoder = _create_encoders(encoder_class,
                                                                         hyper_params)

        if bert_question_encoder.post_pooling_last_hidden_size != \
                bert_paragraph_encoder.post_pooling_last_hidden_size:
            raise ValueError("question/paragraph encoder should have the same output hidden size")
        previous_hidden_size = bert_question_encoder.post_pooling_last_hidden_size
        model = FeedForwardRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], debug,
            hyper_params['model'], previous_hidden_size=previous_hidden_size)
    elif hyper_params['model']['name'] == 'polyencoder':
        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder_class = CachedBertEncoder
        elif re.search('bart', hyper_params['model'].get('bert_base')):
            if hyper_params['model'].get('encoder_only', None):
                encoder_class = EncoderBartEncoder
            else:
                encoder_class = BartAsEncoder
        else:
            encoder_class = BertAsEncoder

        bert_paragraph_encoder, bert_question_encoder = _create_polyencoders(encoder_class,
                                                                         hyper_params)
        
        model = PolyEncoderRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], hyper_params['model'], debug)
    elif hyper_params['model']['name'] == 'inverse_polyencoder':
        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder_class = CachedBertEncoder
        elif 't5' in hyper_params['model'].get('bert_base'):
            encoder_class = T5EncAsEncoder
        elif re.search('bart', hyper_params['model'].get('bert_base')):
            if hyper_params['model'].get('single_decoder', None):
                if hyper_params['model'].get('encoder_only', None):
                    encoder_class = EncoderBartEncoder
                else:
                    encoder_class = BartAsEncoder
            else:
                encoder_class = DualDecBart
        else:
            encoder_class = BertAsEncoder

        bert_paragraph_encoder, bert_question_encoder = _create_inverse_polyencoders(encoder_class,
                                                                         hyper_params)
        
        model = InversePolyEncoderRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], hyper_params['model'], debug)
    else:
        raise ValueError('model name {} not supported'.format(hyper_params['model']['name']))
    return model


def _create_encoders(encoder, hyper_params):
    if hyper_params['model']['single_encoder']:
        logger.info('using a single BERT for both questions and answers')
        bert_question_encoder = encoder(
            hyper_params, bert_model=None, name='question')
        bert_paragraph_encoder = encoder(
            hyper_params, bert_model=bert_question_encoder.bert, name='paragraph')
    else:
        logger.info('using 2 BERT models: one for questions and one for answers')
        bert_question_encoder = encoder(hyper_params, bert_model=None, name='question')
        bert_paragraph_encoder = encoder(hyper_params, bert_model=None, name='paragraph')
    return bert_paragraph_encoder, bert_question_encoder

def _create_polyencoders(encoder, hyper_params):
    temp = hyper_params['model']['pooling_type']
    hyper_params['model']['pooling_type'] = 'polyencoder_context'
    if hyper_params['model']['single_encoder']:
        logger.info('using a single BERT for both questions and answers')
        bert_question_encoder = encoder(
            hyper_params, bert_model=None, name='question')
        hyper_params['model']['pooling_type'] = temp
        bert_paragraph_encoder = encoder(
            hyper_params, bert_model=bert_question_encoder.bert, name='paragraph')
    else:
        logger.info('using 2 BERT models: one for questions and one for answers')
        bert_question_encoder = encoder(hyper_params, bert_model=None, name='question')
        hyper_params['model']['pooling_type'] = temp
        bert_paragraph_encoder = encoder(hyper_params, bert_model=None, name='paragraph')
    return bert_paragraph_encoder, bert_question_encoder

def _create_inverse_polyencoders(encoder, hyper_params):
    temp = hyper_params['model']['pooling_type']
    if hyper_params['model']['single_encoder']:
        logger.info('using a single BERT for both questions and answers')
        bert_question_encoder = encoder(
            hyper_params, bert_model=None, name='question')
        hyper_params['model']['pooling_type'] = 'polyencoder_context'
        bert_paragraph_encoder = encoder(
            hyper_params, bert_model=bert_question_encoder.bert, name='paragraph')
    else:
        logger.info('using 2 BERT models: one for questions and one for answers')
        bert_question_encoder = encoder(hyper_params, bert_model=None, name='question')
        hyper_params['model']['pooling_type'] = 'polyencoder_context'
        bert_paragraph_encoder = encoder(hyper_params, bert_model=None, name='paragraph')
    hyper_params['model']['pooling_type'] = temp
    return bert_paragraph_encoder, bert_question_encoder
