"""
Script used to add out-of-distribution examples in a data file.
It basically merges tw ofiles: #1 containing in-distrubution examples,
and #2 containing out-of-distribution examples.
"""
import argparse
import json
import logging

from bert_reranker.data.data_loader import get_passages_by_source, get_passage_id, \
    is_in_distribution

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", help="main json file", required=True)
    parser.add_argument("--input-ood", help="json file where to extract the ood", required=True)
    parser.add_argument("--output", help="output file", required=True)
    parser.add_argument("--max-ood", help="max amount of ood to use. -1 means all of them",
                        type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # parse input data - check there is no OOD
    with open(args.input, 'r', encoding='utf-8') as in_stream:
        input_data = json.load(in_stream)

    _, pid2passages, _ = get_passages_by_source(input_data)
    for example in input_data['examples']:
        related_passage_pid = get_passage_id(example)
        related_passage = pid2passages[related_passage_pid]
        if not is_in_distribution(related_passage):
            raise ValueError(
                '--input file must not have any OOD -- see example {} / passage {}'.format(
                    example['id'], related_passage_pid))

    # new pid number (for OOD) is the highest one plus 1 (so there is no collision)
    new_pid_for_ood = max(pid2passages.keys()) + 1
    # new id number for example is the highest one plus 1 (so there is no collision)
    highest_id_for_example = max([ex['id'] for ex in input_data['examples']]) + 1

    result = input_data

    with open(args.input_ood, 'r', encoding='utf-8') as in_stream:
        ood_data = json.load(in_stream)

    new_ood_passage = {
            "passage_id": new_pid_for_ood,
            "source": "ood_source",
            "uri": None,
            "reference_type": "ood",
            "reference": {
                "page_title": None,
                "section_headers": [],
                "section_content": None,
                "selected_span": None
            }
        }
    result['passages'].append(new_ood_passage)

    _, pid2passages, _ = get_passages_by_source(ood_data)
    id_count = 0
    added_ood = 0
    for example in ood_data['examples']:
        related_passage = get_passage_id(example)
        related_passage = pid2passages[related_passage]
        if is_in_distribution(related_passage):
            id_count += 1
        else:
            example['passage_id'] = new_pid_for_ood
            example['id'] = highest_id_for_example
            highest_id_for_example += 1
            result['examples'].append(example)
            added_ood += 1
        if args.max_ood > -1 and added_ood >= args.max_ood:
            break

    logger.info('kept {} ood from {} (and skipped {} id from the same file)'.format(
        added_ood, args.input_ood, id_count
    ))

    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump(result, ostream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
