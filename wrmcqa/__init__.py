#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from pathlib import Path

if sys.version_info < (3, 5):
    raise RuntimeError('WRMCQA supports Python 3.5 or higher.')

DATA_DIR = (
    os.getenv('WRMCQA_DATA') or
    os.path.join(Path(__file__).absolute().parents[1].as_posix(), 'data')
)

from . import tokenizers
from . import reader
from . import retriever
from . import pipeline
