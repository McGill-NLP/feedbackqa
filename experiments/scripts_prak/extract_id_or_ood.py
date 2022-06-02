"""
Script used to extract in-distibution and/or out-of-distribution from a data file.
"""
import argparse
import json
import logging

from bert_reranker.data.data_loader import get_passages_by_source, get_examples, get_passage_id, \
    is_in_distribution

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", help="join files to filter", required=True)
    parser.add_argument("--output", help="output file", required=True)
    parser.add_argument("--keep-id", help="will keep id", action="store_true")
    parser.add_argument("--keep-ood", help="will keep ood", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input, 'r', encoding='utf-8') as in_stream:
        input_data = json.load(in_stream)

    filtered_examples = []
    filtered_passage_ids = set()
    _, pid2passages, _ = get_passages_by_source(input_data)

    id_kept = 0
    ood_kept = 0
    total = 0

    for example in get_examples(input_data, True):
        example_pid = get_passage_id(example)
        related_passage = pid2passages[example_pid]
        is_id = is_in_distribution(related_passage)
        if is_id and args.keep_id:
            filtered_examples.append(example)
            filtered_passage_ids.add(example_pid)
            id_kept += 1
        elif not is_id and args.keep_ood:
            filtered_examples.append(example)
            filtered_passage_ids.add(example_pid)
            ood_kept += 1
        total += 1

    filtered_passages = [pid2passages[pid] for pid in filtered_passage_ids]

    logger.info('kept {} ID and {} OOD (from a total of {} examples)'.format(
        id_kept, ood_kept, total))
    logger.info('kept {} passages (from a total of {} passages)'.format(
        len(filtered_passages), len(pid2passages)))

    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump({'examples': filtered_examples, 'passages': filtered_passages}, ostream, indent=4,
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
