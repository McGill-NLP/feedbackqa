import glob
import logging
import os
import re

logger = logging.getLogger(__name__)

VAL_ACC_EXTRACTOR = re.compile('val_acc_0=-?\\d+.\\d+')


def _get_accuracy_from_name(model_file):
    re_result = VAL_ACC_EXTRACTOR.findall(model_file)[0]
    return float(re_result.split('=')[1])


def try_to_restore_model_weights(folder):
    model_files = glob.glob(os.path.join(folder, '*.ckpt'))

    if not model_files:
        logger.info('no model found to restore')
        return
    logger.info('found the following model files\n{}'.format('\n'.join(model_files)))
    model_to_acc = {_get_accuracy_from_name(x): x for x in model_files}
    worst_to_best = sorted(list(model_to_acc.keys()))
    best_model = model_to_acc[worst_to_best[-1]]
    logger.info('restoring best model (acc_val-wise): {}'.format(best_model))

    return best_model
