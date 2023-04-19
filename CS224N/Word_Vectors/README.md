# Exploring Word Embeddings

<div>

_The first project is a warmup that includes a short script focused on calculating co-occurrence matrices._

## Code and Running the Tester

All code is entered into `src/run.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

`$ python tester.py`

## How to build a co-occurrence based word embedding model

Word Vectors are often used as a fundamental component for downstream NLP tasks, e.g. question answering, text generation, translation, etc., so it is important to build some intuitions as to their strengths and weaknesses. Here we explore two types of word vectors: those derived from co-occurrence matrices (following section), and those derived via word2vec (last section - `TBD`).

**Note on Terminology:** The terms `\word vector` and `\word embedding` are often used interchangeably. The term `\embedding` refers to the fact that we are encoding aspects of a word's meaning in a lower dimensional space. As Wikipedia states, "_\conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension_".

### Computing Co-occurence Embeddings

In this project we implement a co-occurrence based word embedding model. We use the Reuters (business and financial news) corpus. This corpus consists of 10,788 news documents totaling 1.3 million words. These documents span 90 categories and are split into train and test sets. For more details, please [see](https://www.nltk.org/book/ch02.html).
