# Neural Transition-Based Dependency Parsing

<div>

In this project, we build a neural dependency parser using PyTorch. We implement a NN based dependency parser, with the goal of maximizing performance on the UAS (Unlabeled Attachment Score) metric.

This project requires PyTorch without CUDA installed. GPUs

A dependency parser analyzes the grammatical structure of a sentence establishing relationships between head words, and words which modify those heads. This implementation is a _transition-based_ parser, which incrementally builds up a parse one step at a time. At every step it maintains a _partial parse_, which is represented as follows:

- A _stack_ of words that are currently being processed.
- A _buffer_ of words yet to be processed.
- A list of _dependencies_ predicted by the parser.

Initially, the stack only contains ROOT, the dependencies list is empty and the buffer contains all words of the sentence in order. At each step, the parser applies a _transition_ to the partial parse until its buffer is empty and the stack size is 1. The following transitions can be applied:

- `SHIFT`: removes the first word from the buffer and pushes it onto the stack.
- `LEFT-ARC`: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack.
- `RIGHT-ARC`: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack.

On each step, your parser will decide among the three transitions using a neural network classifier.

## Code and Running the Tester 🛠️

All code is entered into `src/run.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

```bash
$ python tester.py
```
