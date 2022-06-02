import logging
import os

from pytorch_lightning import Callback


logger = logging.getLogger(__name__)


class CacheManagerCallback(Callback):

    def __init__(self, retriever, output_folder):
        self.q_encoder = retriever.bert_question_encoder
        self.p_encoder = retriever.bert_paragraph_encoder
        self.output_folder = output_folder

    def on_init_start(self, trainer):
        q_path = os.path.join(self.output_folder, 'qcache.pkl')
        if os.path.exists(q_path):
            logger.info('loading cache for questions from {}'.format(q_path))
            loaded = self.q_encoder.load_cache(q_path)
            logger.info('loaded {} entries'.format(loaded))
        p_path = os.path.join(self.output_folder, 'pcache.pkl')
        if os.path.exists(p_path):
            logger.info('loading cache for paragraphs from {}'.format(p_path))
            loaded = self.p_encoder.load_cache(p_path)
            logger.info('loaded {} entries'.format(loaded))

    def on_epoch_end(self, trainer, pl_module):
        self.q_encoder.save_cache(os.path.join(self.output_folder, 'qcache.pkl'))
        self.p_encoder.save_cache(os.path.join(self.output_folder, 'pcache.pkl'))
        self.q_encoder.print_stats_to(logger.info)
        self.p_encoder.print_stats_to(logger.info)
