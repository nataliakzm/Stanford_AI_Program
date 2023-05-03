# Text Reconstruction 📝

<div>

Here we delve into two tasks: _word segmentation_ and _vowel insertion_, which are relevant in non-English languages, where words might not be flanked by spaces, or vowels are often not written.

## Project Overview

**Word segmentation** is a common challenge in languages where words aren't typically separated by spaces, such as Chinese, or in situations with long compound German words. Given a continuous string of alphabetical characters (`[a-z]`), the goal is to insert spaces into this string to generate a sentence that is the most fluent according to the LM.

On the other hand, **vowel insertion** becomes necessary in languages like Arabic or Hebrew, where modern script often omits vowel sounds and leaves their inference to the reader. In this project, we consider a sequence of English words with their vowels `(A, E, I, O, and U, never Y)` missing. The task is to place vowels back into these words in a way that maximizes sentence fluency, i.e., minimizes sentence cost. We approach this task with a bigram cost function.
 
These tasks are great examples of modeling, where we convert real-world tasks into state-space search problems. The overarching aim is to develop a system capable of reconstructing understandable and fluent sentences from incomplete textual data.

## Code and Running the Tester

All code is entered into `src/model.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

```bash
$ python tester.py
```
