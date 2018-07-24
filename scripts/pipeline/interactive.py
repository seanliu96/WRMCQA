#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive interface to full WRMCQA pipeline."""

import torch
import argparse
import code
import prettytable
import logging

from termcolor import colored
from wrmcqa import pipeline
from wrmcqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

WRMCQA = None

# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(question, candidates=None, top_n=1, n_docs=5):
    predictions = WRMCQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc', 'Overall Score', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p['span'], p['doc_id'],
                       '%.5g' % p['overall_score'], 
                       '%.5g' % p['span_score'],
                       '%.5g' % p['doc_score']])
    print('Top Predictions:')
    print(table)
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        start = p['context']['start']
        end = p['context']['end']
        output = (text[:start] +
                  colored(text[start: end], 'green', attrs=['bold']) +
                  text[end:])
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')


banner = """
* WRMCQA *

* Author: Xin Liu, undergraduate of SDCS, SYSU

* Implement based on Facebook's DrQA

>> process(question, candidates=None, top_n=1, n_docs=5)
>> usage()
"""

def usage():
    print(banner)

# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-model', type=str, default=None,
                        help='Path to trained Document Reader model')
    parser.add_argument('--retriever-model', type=str, default=None,
                        help='Path to Document Retriever model (tfidf)')
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--embedding-file', type=str, default=None,
                        help=("Expand dictionary to use all pretrained "
                            "embeddings in this file"))
    parser.add_argument('--char-embedding-file', type=str, default=None,
                        help=("Expand char dictionary to use all pretrained "
                            "embeddings in this file"))
    parser.add_argument('--tokenizer', type=str, default=None,
                        help=("String option specifying tokenizer type to "
                            "use (e.g. 'spacy')"))
    parser.add_argument('--candidate-file', type=str, default=None,
                        help=("List of candidates to restrict predictions to, "
                            "one candidate per line"))
    parser.add_argument('--no-cuda', action='store_true',
                        help="Use CPU only")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="Specify GPU device id to use")
    parser.add_argument('--use-ala', action='store_true',
                        help='Use Answer Ranking Algorithm')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    if args.candidate_file:
        logger.info('Loading candidates from %s' % args.candidate_file)
        candidates = set()
        with open(args.candidate_file) as f:
            for line in f:
                line = utils.normalize(line.strip()).lower()
                candidates.add(line)
        logger.info('Loaded %d candidates.' % len(candidates))
    else:
        candidates = None

    logger.info('Initializing pipeline...')
    WRMCQA = pipeline.WRMCQA(
        reader_model=args.reader_model,
        embedding_file=args.embedding_file,
        char_embedding_file=args.char_embedding_file,
        tokenizer=args.tokenizer,
        fixed_candidates=candidates,
        cuda=args.cuda,
        db_config={'options': {'db_path': args.doc_db}},
        ranker_config={'options': {'tfidf_path': args.retriever_model}},
        use_ala=args.use_ala
    )
    code.interact(banner=banner, local=locals())
