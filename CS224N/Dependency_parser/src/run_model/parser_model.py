#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parser_utils import load_and_preprocess_data


class ParserModel(nn.Module):
    """ Feedforward NN with an embedding layer and single hidden layer.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.

    PyTorch Notes:
        - Note that "ParserModel" is a subclass of the "nn.Module" class. In PyTorch all neural networks
            are a subclass of this "nn.Module".
        - The "__init__" method is where you define all the layers and their respective parameters
            (embedding layers, linear layers, dropout layers, etc.).
        - "__init__" gets automatically called when you create a new instance of your class, e.g.
            when you write "m = ParserModel()".
        - Other methods of ParserModel can access variables that have "self." prefix.  
    """

    def __init__(self, embeddings, n_features=36,
                 hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (Tensor): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        torch.manual_seed(0)
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))

        self.embed_to_hidden = nn.Linear(self.embed_size * self.n_features, self.hidden_size, bias = True)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        
        self.dropout = nn.Dropout(p = self.dropout_prob)

        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes, bias = True)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)

    def embedding_lookup(self, t):
        """ Utilize `self.pretrained_embeddings` to map input `t` from input tokens (integers)
            to embedding vectors.

            PyTorch Notes:
                - `self.pretrained_embeddings` is a torch.nn.Embedding object that we defined in __init__
                - Here `t` is a tensor where each row represents a list of features. Each feature is represented by an integer (input token).
                - In PyTorch the Embedding object, e.g. `self.pretrained_embeddings`, allows you to
                    go from an index to embedding.  

            @param t (Tensor): input tensor of tokens (batch_size, n_features)
            @return x (Tensor): tensor of embeddings for words represented in t
                                (batch_size, n_features * embed_size)
        """

        x = self.pretrained_embeddings(t) # (batch_size, n_features, embed_size)
        x = x.view(x.size()[0], -1) # (batch_size, n_features * embed_size)
        return x

    def forward(self, t):
        """ Run the model forward.
            PyTorch Notes:
                - Every nn.Module object (PyTorch model) has a `forward` function.
                - When you apply your nn.Module to an input tensor `t` this function is applied to the tensor.
                    For example, if you created an instance of your ParserModel and applied it to some `t` as follows,
                    the `forward` function would called on `t` and the result would be stored in the `output` variable:
                        model = ParserModel()
                        output = model(t) # this calls the forward function

        @param t (Tensor): input tensor of tokens (batch_size, n_features)
        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """

        x = self.embedding_lookup(t)
        h = F.relu(self.embed_to_hidden(x))
        
        logits = self.hidden_to_logits(self.dropout(h))
        return logits
