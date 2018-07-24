# WRMCQA
This is [Xin Liu](https://www.linkedin.com/in/xin-liu-179830143/)'s final year project, Wikipedia Retrieval and Machine Comprehension Question Answering (WRMCQA), based on Fackbook's [DrQA](https://github.com/facebookresearch/DrQA). WRMCQA belongs to [HKUST-KnowComp](https://github.com/HKUST-KnowComp) and is under the [BSD LICENSE](LICENSE).

## Quick Links

- [What is MRMCQA](#What is WRMCQA)
- [Components](#Components)
- [Installation](#Installation)
- [Quick Start: Demo](#Quick Start: Demo)

## What is WRMCQA

WRMCQA is an open-domain text-based question answering system using Information Retrieval and Machine Comprehension. It follows the idea of DrQA and introduces a useful algorithm to scale this Machine Comprehension Module to long documents rather than a short paragraph. In particular, WRMCQA retrieves several relavent unstructured documents and uses MC models to find answers in paragraphs.  It is the ranking algorithm that determines the final answer at last. 

Experiments with WRMCQA focus on answering factoid questions while using Wikipedia as the unique knowledge source for documents. Wikipedia, Baidu Baike and other encyclopedias are human collaboratively constructed text collections that contain information collected in traditional encyclopedias and news related to recent events. These encyclopedias are ideal sources. Here WRMCQA treats Wikipedia as a generic collection of articles and does not rely on its internal graph structure. In fact, WRMCQA can be applied to any collection of documents, as described in the retriever [README](scripts/retriever/README.md).

This repository includes code, data and pre-trained models.

## Components

### Document Retriever

**Document Retriever is one part of DrQA, so the following is from [DrQA's README](https://github.com/facebookresearch/DrQA#drqa-components)**

DrQA is not tied to any specific type of retrieval system -- as long as it effectively narrows the search space and focuses on relevant documents.

Following classical QA systems, we include an efficient (non-machine learning) document retrieval system based on sparse, TF-IDF weighted bag-of-word vectors. We use bags of hashed n-grams (here, unigrams and bigrams).

To see how to build your own such model on new documents, see the retriever [README](scripts/retriever/README.md).

To interactively query Wikipedia:

```bash
python scripts/retriever/interactive.py --model /path/to/model
```

If `model` is left out our [default model](#Stochastic Mnemonic Reader) will be used (assuming it was [downloaded](#Installation)).

To evaluate the retriever accuracy (% match in top 5) on a dataset:

```bash
python scripts/retriever/eval.py /path/to/format/A/dataset.txt --model /path/to/model
```

### Stochastic Mnemonic Reader (SM Reader)

WRMCQA's Stochastic Mnemonic Reader is a multi-layer recurrent neural network machine comprehension model trained to do extractive question answering. That is, the model tries to find an answer to any question as a text span in one of the returned documents.

The Stochastic Mnemonic Reader is the core of the QA system and it is very complex. It combines a multi-layer BiLSTM with Maxout, the Dot-product Attention, Senmantic Fusion Units, the Bilinear Term, the Answer pointer with Stochastic Dropout. 

The reader was primarily trained on the [SQuAD](https://arxiv.org/abs/1606.05250) dataset. But it can be used standalone on such SQuAD-like tasks where a specific context is supplied with the question, the answer to which is contained in the context.

To see how to train the Stochastic Mnemonic Reader or other readers on SQuAD, such as the RNN Reader, the R-Net Reader and the Mnemonic Reader,  see the reader [README](scripts/reader/README.md).

To interactively ask questions about text with a trained model:

```bash
python scripts/reader/interactive.py --model /path/to/model
```

Again, here `model` is optional; a default model will be used if it is left out.

To run model predictions on a dataset:

```bash
python scripts/reader/predict.py /path/to/format/B/dataset.json --model /path/to/model
```

### Answer Ranking Algorithm (ALA)

After get answers from the Stochastic Mnemonic Reader, an algorithm is needed to determine which one is  better to answer the question. If you need it, remember to add it to parameters explicitly.

### Pipeline

The full system is linked together in `wrmcqa.pipeline.WRMCQA`.

To interactively ask questions using the full WRMCQA:

```bash
python scripts/pipeline/interactive.py
```

Optional arguments:

```
--reader-model    Path to trained Document Reader model.
--retriever-model Path to Document Retriever model (tfidf).
--doc-db          Path to Document DB.
--tokenizer       String option specifying tokenizer type to use (e.g. 'spacy').
--candidate-file  List of candidates to restrict predictions to, one candidate per line.
--no-cuda         Use CPU only.
--gpu             Specify GPU device id to use.
--use-ala         Use Answer Ranking Algorithm.
```

To run predictions on a dataset:

```bash
python scripts/pipeline/predict.py /path/to/format/A/dataset.txt
```

Optional arguments:

```
--out-dir             Directory to write prediction file to (<dataset>-<model>-pipeline.preds).
--reader-model        Path to trained Document Reader model.
--retriever-model     Path to Document Retriever model (tfidf).
--doc-db              Path to Document DB.
--embedding-file      Expand dictionary to use all pretrained embeddings in this file (e.g. all glove vectors to minimize UNKs at test time).
--candidate-file      List of candidates to restrict predictions to, one candidate per line.
--n-docs              Number of docs to retrieve per query.
--top-n               Number of predictions to make per query.
--tokenizer           String option specifying tokenizer type to use (e.g. 'spacy').
--no-cuda             Use CPU only.
--gpu                 Specify GPU device id to use.
--parallel            Use data parallel (split across GPU devices).
--use-ala             Use Answer Ranking Algorithm.
--num-workers         Number of CPU processes (for tokenizing, etc).
--batch-size          Document paragraph batching size (Reduce in case of GPU OOM).
--predict-batch-size  Question batching size (Reduce in case of CPU OOM).
```

### Distant Supervision (DS)

WRMCQA's performance improves significantly in the full-setting when provided with distantly supervised data from additional datasets. Given question-answer pairs but no supporting context, we can use string matching heuristics to automatically associate paragraphs to these training examples.

> Question: What U.S. state’s motto is “Live free or Die”?
>
> Answer: New Hampshire
>
> DS Document: Live Free or Die
> **“Live Free or Die”** is the official **motto** of the **U.S. state** of _**New Hampshire**_, adopted by the **state** in 1945. It is possibly the best-known of all state mottos, partly because it conveys an assertive independence historically found in American political philosophy and partly because of its contrast to the milder sentiments found in other state mottos.

The `scripts/distant` directory contains code to generate and inspect such distantly supervised data. More information can be found in the distant supervision [README](scripts/distant/README.md).

### Tokenizers

We provide a number of different tokenizer options for convenience. Each has its own pros/cons based on how many dependencies it requires, overhead for running it, speed, and performance. For our reported experiments we used spaCy (but results are all similar).

Available tokenizers:

- _SpacyTokenizer_: Uses [spaCy](https://spacy.io/) (option: 'spacy', default).
- _CoreNLPTokenizer_: Uses [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) (option: 'corenlp').
- _RegexpTokenizer_: Custom regex-based PTB-style tokenizer (option: 'regexp').
- _SimpleTokenizer_: Basic alpha-numeric/non-whitespace tokenizer (option: 'simple').

See the [list](wrmcqa/tokenizers/__init__.py) of mappings between string option names and tokenizer classes.

## Installation

WRMCQA supports Windows/Linux/OSX and requires Python 3.5 or higher. It also requires installing [PyTorch](http://pytorch.org/). Its other dependencies are listed in [requirements.txt](requirements.txt). CUDA is strongly recommended for speed, but not necessary.

Run the following commands to clone the repository and install WRMCQA:

```bash
git clone git@github.com:HKUST-KnowComp/WRMCQA.git
cd WRMCQA
pip install -r requirements.txt
python setup.py develop
```

Note: requirements.txt includes a subset of all the possible required packages. 

If you prefer to use the SpacyTokenizer, you also need to download spaCy `en` model. 

Else if you want to use Stanford CoreNLP, you have to download the Stanford CoreNLP jars via [install_corenlp.sh](install_corenlp.sh) and have the jars in your java `CLASSPATH` environment variable, or set the path programmatically with:

```python
import wrmcqa.tokenizers
wrmcqa.tokenizers.set_default('corenlp_classpath', '/your/corenlp/classpath/*')
```

For convenience, the Document Reader, Retriever, and Pipeline modules will try to load default models if no model argument is given. See below for downloading these models.

### Trained Models and Data

To download all provided trained models and data for Wikipedia question answering, run:

```bash
./download.sh
```

_Warning: this downloads a 7.5GB tarball (25GB untarred) and will take some time._

This stores the data in `data/` at the file paths specified in the various modules' defaults. This top-level directory can be modified by setting a `WRMCQA_DATA` environment variable to point to somewhere else.

Default directory structure (see [embeddings](scripts/reader/README.md#note-on-word-embeddings) for more info on additional downloads for training):

```
WRMCQA
├── data (or $WRMCQA_DATA)
    ├── datasets
    │   ├── SQuAD-v1.1-<train/dev>.<txt/json>
    │   ├── CuratedTrec-<train/test>.txt
    │   └── WikiMovies-<train/test/entities>.txt
    ├── reader
    │   ├── m_reader.mdl
    │   ├── r_net_reader.mdl
    │   ├── rnn_reader.mdl
    │   └── sm_reader.mdl
    └── wikipedia
        ├── docs.db
        └── docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
```

Default model paths for the different modules can also be modified programmatically in the code, e.g.:

```python
import wrmcqa.reader
wrmcqa.reader.set_default('model', '/path/to/model')
reader = wrmcqa.reader.Predictor()  # Default model loaded for prediction
```

#### Document Retriever

TF-IDF model using Wikipedia (unigrams and bigrams, 2^24 bins, simple tokenization), evaluated on multiple datasets (test sets, dev set for SQuAD):

|                            Model                             | SQuAD P@5 | CuratedTREC P@5 | WebQuestions P@5 | WikiMovies P@5 | Size  |
| :----------------------------------------------------------: | :-------: | :-------------: | :--------------: | :------------: | :---: |
| [TF-IDF model](https://s3.amazonaws.com/fair-data/drqa/docs-tfidf-ngram%3D2-hash%3D16777216-tokenizer%3Dsimple.npz.gz) |   78.0    |      87.6       |       75.0       |      69.8      | ~13GB |

_P@5 here is defined as the % of questions for which the answer segment appears in one of the top 5 documents_.

#### Stochastic Mnemonic Reader

Models trained only on SQuAD, evaluated in the SQuAD setting:

|    Model     | SQuAD Dev EM | SQuAD Dev F1 | Size |
| :----------: | :----------: | :----------: | :--: |
|  rnn_reader  |     69.4     |     78.6     | 118M |
| r_net_reader |     69.9     |     79.2     | 122M |
|   m_reader   |     72.5     |     81.1     | 120M |
|  sm_reader   |   **73.2**   |   **81.8**   | 122M |

\* The rnn_reader is the open-source version of DrQA's Document Reader.

\* The r_net_reader is implemented by the author according to [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). It uses BiLSTM rather than BiGRU because of one bug of Pytotch.

\* The m_reader is implemented by the author according to [Reinforced Mnemonic Reader for Machine Comprehension](https://arxiv.org/pdf/1705.02798.pdf).

![EM_F1](img/EM_F1.png)



Models **without NER/POS/lemma/TF features**, evaluated on multiple datasets (test sets, dev set for SQuAD) in the full Wikipedia setting:

|    Model    | SQuAD EM  | CuratedTrec EM | WikiMovies EM | Size |
| :---------: | :-------: | :------------: | :-----------: | :--: |
|  sm_reader  |   19.56   |     15.99      |     19.54     | 130M |
|    + ALA    |   22.02   |     19.88      |     21.00     |  -   |
|    + DS2    |   30.15   |     22.77      |     34.54     | 130M |
| + DS2 + ALA | **30.68** |   **27.09**    |     35.78     |  -   |
|     DS3     |   26.66   |     22.62      |     35.71     | 244M |
|  DS3 + ALA  |   27.79   |     26.08      |   **36.67**   |  -   |

#### Wikipedia

The full-scale experiments were conducted on the 2016-12-21 dump of English Wikipedia. The dump was processed with the [WikiExtractor](https://github.com/attardi/wikiextractor) and filtered for internal disambiguation, list, index, and outline pages (pages that are typically just links). Documents are stored in an sqlite database for which `wrmcqa.retriever.DocDB` provides an interface.

|                           Database                           | Num. Documents | Size |
| :----------------------------------------------------------: | :------------: | :--: |
| [Wikipedia](https://s3.amazonaws.com/fair-data/drqa/docs.db.gz) |   5,075,182    | 13GB |

#### QA Datasets

The datasets used for WRMCQA training and evaluation can be found here:

- SQuAD: [train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- WikiMovies: [train/test/entities](https://s3.amazonaws.com/fair-data/drqa/WikiMovies.tar.gz)
  (Rehosted in expected format from https://research.fb.com/downloads/babi/)
- CuratedTrec: [train/test](https://s3.amazonaws.com/fair-data/drqa/CuratedTrec.tar.gz)
  (Rehosted in expected format from https://github.com/brmson/dataset-factoid-curated)

##### Format A

The `retriever/eval.py`, `pipeline/eval.py`, and `distant/generate.py` scripts expect the datasets as a `.txt` file where each line is a JSON encoded QA pair, like so:

```python
'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'
```

Scripts to convert SQuAD to this format are included in `scripts/convert`. This is automatically done in `download.sh`.

##### Format B

The `reader` directory scripts expect the datasets as a `.json` file where the data is arranged like SQuAD:

```
file.json
├── "data"
│   └── [i]
│       ├── "paragraphs"
│       │   └── [j]
│       │       ├── "context": "paragraph text"
│       │       └── "qas"
│       │           └── [k]
│       │               ├── "answers"
│       │               │   └── [l]
│       │               │       ├── "answer_start": N
│       │               │       └── "text": "answer"
│       │               ├── "id": "<uuid>"
│       │               └── "question": "paragraph question?"
│       └── "title": "document id"
└── "version": 1.1
```

##### Entity lists

Some datasets have (potentially large) candidate lists for selecting answers. For example, WikiMovies' answers are OMDb entries. If candidates have been known , we can impose that all predicted answers must be in this list by discarding any higher scoring spans that are not.

## Quick Start: Demo

[Install](#Installation) WRMCQA and [download](#Trained Models and Data) our models to start asking open-domain questions!

Run `python scripts/pipeline/interactive.py` to drop into an interactive session. For each question, the top span and the Wikipedia paragraph it came from are returned.

```
>>> process('What is qa')
04/13/2018 10:56:36 PM: [ Processing 1 queries... ]
04/13/2018 10:56:36 PM: [ Retrieving top 5 docs... ]
04/13/2018 10:56:36 PM: [ Reading 205 paragraphs... ]
04/13/2018 10:56:37 PM: [ Processed 1 queries in 1.1931 (s) ]
Top Predictions:
+------+----------------------------------------------------------------------------------------------------------+--------------------+---------------+--------------+-----------+
| Rank |                                                  Answer                                                  |        Doc         | Overall Score | Answer Score | Doc Score |
+------+----------------------------------------------------------------------------------------------------------+--------------------+---------------+--------------+-----------+
|  1   | a computer science discipline within the fields of information retrieval and natural language processing | Question answering |   2.6957e+05  |    815.83    |   165.21  |
+------+----------------------------------------------------------------------------------------------------------+--------------------+---------------+--------------+-----------+

Contexts:
[ Doc = Question answering ]
Question Answering (QA) is a computer science discipline within the fields of information retrieval and natural language processing (NLP), which is concerned with building systems that automatically answer questions posed by humans in a natural language.
```

```
>>> process('What is the answer to life, the universe, and everything?')
04/13/2018 10:58:19 PM: [ Processing 1 queries... ]
04/13/2018 10:58:19 PM: [ Retrieving top 5 docs... ]
04/13/2018 10:58:19 PM: [ Reading 1460 paragraphs... ]
04/13/2018 10:58:20 PM: [ Processed 1 queries in 1.4437 (s) ]
Top Predictions:
+------+--------+------------------------------------------------+---------------+--------------+-----------+
| Rank | Answer |                      Doc                       | Overall Score | Answer Score | Doc Score |
+------+--------+------------------------------------------------+---------------+--------------+-----------+
|  1   |   42   | Places in The Hitchhiker's Guide to the Galaxy |   2.0026e+07  |    64456     |   138.19  |
+------+--------+------------------------------------------------+---------------+--------------+-----------+

Contexts:
[ Doc = Places in The Hitchhiker's Guide to the Galaxy ]
Although often mistaken for a planet, Earth is in reality the greatest supercomputer of all time, designed by the second greatest supercomputer of all time, Deep Thought, to calculate the Ultimate Question of Life, The Universe and Everything (to which the answer is 42). It was built by the then-thriving custom planet industry of Magrathea to run a ten-million-year program in which organic life would play a major role. Slartibartfast, a Magrathean designer, was involved in the project and signed his name among the fjords of Norway (an area that won him an award).
```

## License
WRMCQA is under [BSD LICENSE](LICENSE).