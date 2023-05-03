# CS224N: NLP with Deep Learning 🚀

<div>

This repository contains ptojects created as part of the Stanford's CS224N: NLP with Deep Learning course. 

Throughout the course, I overcame several challenges that deepened my understanding of NLP, which included:

- Designing, implementing, and comprehending NLP neural network models using the PyTorch framework;
- Representing word meanings using word vectors such as Word2Vec, SVD, and GloVe;
- Identifying semantic relationships between words in a sentence through dependency parsing;
- Making large-scale word predictions using LMs, RNNs, and neural machine translation;
- Improving NLP models and pretraining transformers for more efficient NLP & NLU.

## Getting Started 🧭

### Installation

To get started with the course materials, clone this repository to your local machine:

```bash
$ git clone https://github.com/nataliakzm/Stanford_AI_Program.git
```

Next, navigate into the cloned repository and create a new conda environment from the `environment.yml` file, by running:

```bash
cd Stanford_AI_Program/CS224N/<project>
`$ conda env create -f src/environment.yml
```

Once you've set up your environment, activate it:

```bash
`$ conda activate [your-environment-name]
```

### Usage

Each project from the course is located in its own directory within the repository. To get started with a project, navigate into the corresponding directory and follow the instructions in the README.

- **Project 1** – [**Exploring Word Vectors**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS224N/Word_Vectors): This is a preliminary project that centers on the computation of co-occurrence matrices. 
- **Project 2** – [**Word2Vec**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS224N/Word2Vec): The project's objective is to implement Word2Vec and train custom word vectors using SGD.
- **Project 3** – [**Neural Dependency Parser**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS224N/Dependency_parser): This project involves building a transition-based parser that incrementally constructs a parse. Using PyTorch, we implemented a neural network-based dependency parser with the aim of maximizing performance on the UAS (Unlabeled Attachment Score) metric. 
- **Project 4** – [**English to Cherokee NMT System**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS224N/Machine_translator): Here, we focus on building a Neural Machine Translation (NMT) system that can translate Cherokee sentences into English. To accomplish this, we utilize PyTorch and run the model on a GPU to optimize performance. Our approach is based on a Seq2Seq network with attention, which allows the model to capture the dependencies between words in a sentence and generate accurate translations. 
- **Project 5** – [**Self-Attention, Transformers, and Pretraining**](https://github.com/nataliakzm/Transformer_model): This final project delves into the concepts of attention and pretraining, and focuses on training a Transformer to perform a task that requires accessing knowledge about the world that is not explicitly encoded in its training data. We utilize PyTorch on a GPU for efficient computation. The project consists of several components, including implementing self-attention, designing a Transformer architecture, and pretraining the model on a large corpus of text.
