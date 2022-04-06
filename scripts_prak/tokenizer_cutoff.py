"""
Script used to generate some stats (tokenization-wise) on a data file.
It will also provide some examples of tokenization.
Note that more tokenization techniques can be used.
(use --tokenizer-name to specify a tokenizer, and see Hugging Face documentation for a list)
"""
import argparse
import json
import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def count_cutoff_sentences(sentences, tokenizer, max_length):

    n_sentences_cutoff = 0
    for sentence in sentences:
        #  + 2 is to account for [CLS] and [SEP]
        if len(tokenizer.tokenize(sentence)) + 2 > max_length:
            n_sentences_cutoff += 1

    return n_sentences_cutoff


def evaluate_tokenizer_cutoff(file, tokenizer, max_question_length, max_answer_length):
    with open(file, 'r', encoding='utf-8') as in_stream:
        qa_pairs = json.load(in_stream)
    evaluate_tokenizer_cutoff_from_json(qa_pairs, tokenizer, max_question_length,
                                        max_answer_length)


def evaluate_tokenizer_cutoff_from_json(qa_pairs, tokenizer, max_question_length,
                                        max_answer_length):
    """evaluate how much questions are being cutoff based on tokenizer's max length"""
    # Collect all unique questions and answers
    all_questions = []
    all_answers = set()
    for qa_pair in qa_pairs:
        question, answers = qa_pair
        all_questions.append(question)
        all_answers |= set(answers)
    all_answers = list(all_answers)

    # Analyze how much is being cutoff
    cutoff_results_questions = count_cutoff_sentences(all_questions, tokenizer, max_question_length)
    cutoff_results_answers = count_cutoff_sentences(all_answers, tokenizer, max_answer_length)

    logger.info('Max length used for questions: {}'.format(max_question_length))
    logger.info('Number of questions cutoff by tokenizer: {} / {}, ({:3.2f} %)\n'.format(
        cutoff_results_questions,
        len(all_questions),
        cutoff_results_questions / len(all_questions) * 100
        )
    )
    logger.info('Max length used for answers: {}'.format(max_answer_length))
    logger.info('Number of answers cutoff by tokenizer: {} / {}, ({:3.2f} %)\n'.format(
        cutoff_results_answers,
        len(all_answers),
        cutoff_results_answers / len(all_answers) * 100
        )
    )


def evaluate_tokenizer_cutoff_from_txt(sentences, tokenizer, max_sentence_length):
    """evaluate how much questions are being cutoff based on tokenizer's max length"""
    # Collect all unique questions and answers

    # Analyze how much is being cutoff
    cutoff_results_questions = count_cutoff_sentences(sentences, tokenizer, max_sentence_length)

    logger.info('Max length used for questions: {}'.format(max_sentence_length))
    logger.info('Number of questions cutoff by tokenizer: {} / {}, ({:3.2f} %)\n'.format(
        cutoff_results_questions,
        len(sentences),
        cutoff_results_questions / len(sentences) * 100
        )
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',
                        help='data file containing questions and answers', required=True)
    parser.add_argument('--tokenizer-name',
                        help='name of the tokenizer to use', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if args.input.endswith('.json'):
        analyze_json(args, tokenizer)
    elif args.input.endswith('.txt'):
        analyze_txt(args, tokenizer)
    else:
        raise ValueError('only .json or .txt are supported')


def analyze_txt(args, tokenizer):
    with open(args.input, 'r', encoding='utf-8') as in_stream:
        sentences = in_stream.readlines()
    logger.info('tokenization example for the first 5 utterances:')
    for i in range(5):
        sentence = sentences[i]
        logger.info('question {} "{}" => "{}"'.format(
            i, sentence, tokenizer.tokenize(sentence)))
    max_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500, 750]
    for max_length in max_lengths:
        evaluate_tokenizer_cutoff_from_txt(sentences, tokenizer, max_length)


def analyze_json(args, tokenizer):
    with open(args.input, 'r', encoding='utf-8') as in_stream:
        qa_pairs = json.load(in_stream)
    logger.info('tokenization example for the first 5 utterances:')
    for i in range(5):
        question, candidates = qa_pairs[i]
        logger.info('question {} "{}" => "{}"'.format(
            i, question, tokenizer.tokenize(question)))
        logger.info('candidate {} "{}" => "{}"\n'.format(
            i, candidates[0], tokenizer.tokenize(candidates[0])))
    max_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500, 750]
    for max_length in max_lengths:
        evaluate_tokenizer_cutoff_from_json(qa_pairs, tokenizer, max_length, max_length)


if __name__ == '__main__':
    main()
