#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np


class StanfordSentiment:
    def __init__(self, path=None, tablesize=1000000):
        if not path:
            path = "utils/datasets/stanfordSentimentTreebank"

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens

    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/datasetSentences.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                sentences += [[w.lower() for w in splitted]]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    def num_sentences(self):
        if hasattr(self, "_num_sentences") and self._num_sentences:
            return self._num_sentences
        else:
            self._num_sentences = len(self.sentences())
            return self._num_sentences

    def all_sentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        reject_prob = self.reject_prob()
        tokens = self.tokens()
        allsentences = [[w for w in s
                         if 0 >= reject_prob[tokens[w]] or random.random() >= reject_prob[tokens[w]]]
                        for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences

        return self._allsentences

    def get_random_context(self, C=5):
        allsent = self.all_sentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID + 1 < len(sent):
            context += sent[wordID + 1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.get_random_context(C)

    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.num_sentences()
        sentences = self.sentences()
        for i in range(self.num_sentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            sent_labels[i] = labels[dictionary[full_sent]]

        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for i in range(3)]
        with open(self.path + "/datasetSplit.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        return self._split

    def get_random_train_sentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def get_dev_sentences(self):
        return self.get_split_sentences(2)

    def get_test_sentences(self):
        return self.get_split_sentences(1)

    def get_train_sentences(self):
        return self.get_split_sentences(0)

    def get_split_sentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def sample_table(self):
        if hasattr(self, '_sample_table') and self._sample_table is not None:
            return self._sample_table

        n_tokens = len(self.tokens())
        sampling_freq = np.zeros((n_tokens,))
        self.all_sentences()
        i = 0
        for w in range(n_tokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweight
                freq = freq ** 0.75
            else:
                freq = 0.0
            sampling_freq[i] = freq
            i += 1

        sampling_freq /= np.sum(sampling_freq)
        sampling_freq = np.cumsum(sampling_freq) * self.tablesize

        self._sample_table = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > sampling_freq[j]:
                j += 1
            self._sample_table[i] = j

        return self._sample_table

    def reject_prob(self):
        if hasattr(self, '_reject_prob') and self._reject_prob is not None:
            return self._reject_prob

        threshold = 1e-5 * self._wordcount

        n_tokens = len(self.tokens())
        reject_prob = np.zeros((n_tokens,))
        for i in range(n_tokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweight
            reject_prob[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._reject_prob = reject_prob
        return self._reject_prob

    def sample_token_idx(self):
        return self.sample_table()[random.randint(0, self.tablesize - 1)]
