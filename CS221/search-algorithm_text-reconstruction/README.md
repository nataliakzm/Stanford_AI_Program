# Text Reconstruction

<div>

In this project we work on two tasks: _word segmentation_ and _vowel insertion_.

Word segmentation often comes up when processing many non-English languages, in which words might not be flanked by spaces on either end, such as written Chinese or long compound German words. Vowel insertion is relevant for languages like Arabic or Hebrew, where modern script eschews notations for vowel sounds and the human reader infers them from context. More generally, this is an instance of a reconstruction problem with a lossy encoding and some context.

The goal here is modeling - that is, converting real-world tasks into state-space search
problems.

### Word Segmentation

In word segmentation, we have as input a string of alphabetical characters (`[a-z]`) without whitespace, and the goal is to insert spaces into this string such that the result is the most uent according to the language model.

### Vowel Insertion

Here, we have a sequence of English words with their vowels missing (A, E, I, O, and U; never Y). The task is to place vowels back into these words in a way that maximizes sentence fluency (i.e., that minimizes sentence cost). For this task, we have a bigram cost function.

## Code and Running the Tester

All code is entered into `src/model.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

`$ python tester.py`

