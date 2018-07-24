#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf WRMCQA retriever module."""

import argparse
import code
import prettytable
import logging
from wrmcqa import retriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

RANKER = None

# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=1):
    doc_names, doc_scores = RANKER.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)


banner = """
* WRMCQA interactive TF-IDF WRMCQA Retriever *

* Author: Xin Liu

* Implement based on Facebook's DrQA

>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)

# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    logger.info('Initializing ranker...')
    RANKER = retriever.get_class('tfidf')(tfidf_path=args.model)
    code.interact(banner=banner, local=locals())
