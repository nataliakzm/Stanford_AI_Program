#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from testerUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import os
import traceback

import run_model
import sacrebleu
import nltk

import sys
import torch
import torch.nn as nn
import torch.nn.utils
from typing import Dict
from docopt import docopt
from vocab import Vocab
from nltk.translate.bleu_score import corpus_bleu

# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
# CONSTANTS
# ----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0

LARGE_BATCH_SIZE = 32
LARGE_EMBED_SIZE = 256
LARGE_HIDDEN_SIZE = 256
NONZERO_DROPOUT_RATE = 0.3

def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)

    with torch.no_grad():
        model.apply(init_weights)

def sanity_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']  # only append <s> and </s> to the target sentence
        data.append(sent)
    return data

class DummyVocab():
  def __init__(self):
    self.src = {'<pad>': 1, "one": 1, "two": 2}  #len = 3
    self.tgt = {'<pad>': 3, "one": 1, "two": 2, "three": 3, "four": 4}  #len = 5

def setup():
    # Load training data & vocabulary
    train_data_src = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.es', 'src')
    train_data_tgt = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.en', 'tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    for src_sents, tgt_sents in run_model.batch_iter(train_data, batch_size=LARGE_BATCH_SIZE, shuffle=True):
        src_sents = src_sents
        tgt_sents = tgt_sents
        break
    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')
    return src_sents, tgt_sents, vocab

def weight_copy(stu_model, soln_model):
    stu_model.h_projection.weight = soln_model.h_projection.weight
    stu_model.c_projection.weight = soln_model.c_projection.weight
    stu_model.att_projection.weight = soln_model.att_projection.weight
    stu_model.combined_output_projection.weight = soln_model.combined_output_projection.weight
    stu_model.target_vocab_projection.weight = soln_model.target_vocab_projection.weight

    stu_model.encoder.weight_ih_l0 = soln_model.encoder.weight_ih_l0
    stu_model.encoder.weight_hh_l0 = soln_model.encoder.weight_hh_l0
    stu_model.encoder.bias_ih_l0 = soln_model.encoder.bias_ih_l0
    stu_model.encoder.bias_hh_l0 = soln_model.encoder.bias_hh_l0

    stu_model.decoder.weight_ih = soln_model.decoder.weight_ih
    stu_model.decoder.weight_hh = soln_model.decoder.weight_hh
    stu_model.decoder.bias_ih = soln_model.decoder.bias_ih
    stu_model.decoder.bias_hh = soln_model.decoder.bias_hh

def test_encoding_hiddens(source_padded, source_lengths, stu_model, soln_model, vocab):
  # Prep for Test
  weight_copy(stu_model, soln_model)

  enc_hidden, decode_hidden, decode_cell = False, False, False
  with torch.no_grad():
          enc_hiddens_model, init_hidden_model = stu_model.encode(source_padded, source_lengths)
          enc_hiddens, init_hidden = soln_model.encode(source_padded, source_lengths)
          if np.allclose(enc_hiddens.numpy(), enc_hiddens_model.numpy()):
              enc_hidden = True

          if np.allclose(init_hidden[0].numpy(), init_hidden_model[0].numpy(),atol=1e-4):
              decode_hidden = True

          if np.allclose(init_hidden[1].numpy(), init_hidden_model[1].numpy(),atol=1e-4):
              decode_cell = True

  return enc_hidden, decode_hidden, decode_cell

def test_combined_outputs(source_padded, source_lengths, target_padded, stu_model, soln_model, vocab):
  # Prep for Test
  weight_copy(stu_model, soln_model)
  stu_model.step = soln_model.step
  combined_output = False
  with torch.no_grad():
          enc_hiddens, dec_init_state = soln_model.encode(source_padded, source_lengths)
          enc_masks = soln_model.generate_sent_masks(enc_hiddens, source_lengths)

          combined_outputs_model = stu_model.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
          combined_outputs_pred = soln_model.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
          if np.allclose(combined_outputs_model.numpy(), combined_outputs_pred.numpy(), atol=1e-4):
              combined_output = True
  return combined_output

def test_q1f(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks, stu_model, soln_model):
    """ Compares model output to that of model with dummy data.
    """
    # Prep for Test
    weight_copy(stu_model, soln_model)

    dec_hidden_result, dec_state_result, o_t_result, e_t_result = False, False, False, False
    with torch.no_grad():
            dec_state_model, o_t_model, e_t_model =  stu_model.step(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            dec_state, o_t, e_t =  soln_model.step(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            #dec_state_alt, o_t_alt, e_t_alt = alt_soln_nmt.step(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks)

            if np.allclose(dec_state_model[0].numpy(), dec_state[0].numpy(), atol=1e-4):
                dec_hidden_result = True

            if np.allclose(dec_state_model[1].numpy(), dec_state[1].numpy(), atol=1e-4):
                dec_state_result = True

            if np.allclose(o_t_model.numpy(), o_t.numpy(), atol=1e-4): #or np.allclose(o_t_model.numpy(), o_t_alt.numpy(), atol=1e-4):
                # print("o_t_model_numpy: ", o_t_model.numpy())
                # print("o_t_sol_numpy: ", o_t.numpy())
                o_t_result = True

            if np.allclose(e_t_model.numpy(), e_t.numpy(), atol=1e-4):
                e_t_result = True

    return dec_hidden_result, dec_state_result, o_t_result, e_t_result

def bleu(args: Dict[str, str]):
    """ computes belu score
    @param args (Dict): args for file path details
    """
    # test_data_out = run_model.read_corpus(args['TEST_OUTPUT_FILE'], source='tgt')
    # test_data_gold = run_model.read_corpus(args['TEST_GOLD_FILE'], source='tgt')
    # min_len = min(len(test_data_out), len(test_data_gold))

    # bleu_score = corpus_bleu([[ref] for ref in test_data_gold[:min_len]],
    #                          [hyp for hyp in test_data_out[:min_len]])
    # print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    f = open(args['TEST_OUTPUT_FILE'], "r", encoding='utf8') #change path to run_model
    hyps = []
    for sent in f:
      hyps.append(sent[:-1])     # gets rid of the end \n characters
    f.close()

    f = open(args['TEST_GOLD_FILE'], "r", encoding='utf8') #change to our local path
    refs = []
    for sent in f:
      refs.append(sent[:-1])
    f.close()
    bleu_score = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu_score.score

#########
# TESTS #
class Test_1a(GradedTestCase):
  def setUp(self):
    random.seed(35436)
    np.random.seed(4355)

  @graded(is_hidden=True)
  def test_0(self):
    """1a-0-hidden:  pad sentences"""
    sents = [
      x.split() for x in [
        'hi there',
        'hi there homie',
        'how was your day today',
        'pretty good',
        'how about you',
        'solid',
        'did you watch the warriors game',
        'yup, did you see boogie, that was a sweet seventeen points'
      ]
    ]
    pad_token = '<pad>'
    expected = self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol).pad_sents(sents, pad_token)
    model_result = run_model.pad_sents(sents, pad_token)
    self.assertEqual(expected, model_result)

class Test_1b(GradedTestCase):
  def setUp(self):
    self.vocab = DummyVocab()
    self.model_result = run_model.ModelEmbeddings(12, self.vocab)

  @graded()
  def test_0(self):
    """1b-0-basic: Verify correct class"""
    self.assertIsInstance(self.model_result.source, nn.Embedding)
    self.assertIsInstance(self.model_result.target, nn.Embedding)


  @graded()
  def test_1(self):
    """1b-1-basic: Verify correct parameters"""
    self.assertEqual(self.model_result.source.num_embeddings, 3)
    self.assertEqual(self.model_result.source.embedding_dim, 12)
    self.assertEqual(self.model_result.source.padding_idx, 1)

    self.assertEqual(self.model_result.target.num_embeddings, 5)
    self.assertEqual(self.model_result.target.embedding_dim, 12)
    self.assertEqual(self.model_result.target.padding_idx, 3)

class Test_1c(GradedTestCase):
  def setUp(self):
    self.vocab = DummyVocab()
    self.model_result = run_model.NMT(12, 17, self.vocab, dropout_rate=0.34)

  @graded()
  def test_0(self):
    """1c-0-basic: Verify self.encoder is correct """
    self.assertIsInstance(self.model_result.encoder, nn.LSTM)
    self.assertEqual(self.model_result.encoder.input_size, 12)
    self.assertEqual(self.model_result.encoder.hidden_size, 17)
    self.assertEqual(self.model_result.encoder.bidirectional, True)

  @graded()
  def test_1(self):
    """1c-1-basic: Verify self.decoder is correct """
    self.assertIsInstance(self.model_result.decoder, nn.LSTMCell)
    self.assertEqual(self.model_result.decoder.input_size, 12 + 17)
    self.assertEqual(self.model_result.decoder.hidden_size, 17)
    self.assertEqual(self.model_result.decoder.bias, True)

  @graded()
  def test_2(self):
    """1c-2-basic: Verify that self.h_projection, self.c_projection, and self.att_projection are correct """
    self.assertIsInstance(self.model_result.h_projection, nn.Linear)
    self.assertEqual(self.model_result.h_projection.in_features, 2 * 17)
    self.assertEqual(self.model_result.h_projection.out_features, 17)
    self.assertEqual(self.model_result.h_projection.bias, None)

    self.assertIsInstance(self.model_result.c_projection, nn.Linear)
    self.assertEqual(self.model_result.c_projection.in_features, 2 * 17)
    self.assertEqual(self.model_result.c_projection.out_features, 17)
    self.assertEqual(self.model_result.c_projection.bias, None)

    self.assertIsInstance(self.model_result.att_projection, nn.Linear)
    self.assertEqual(self.model_result.att_projection.in_features, 2 * 17)
    self.assertEqual(self.model_result.att_projection.out_features, 17)
    self.assertEqual(self.model_result.att_projection.bias, None)

  @graded()
  def test_3(self):
    """1c-3-basic: Verify that self.combined_output_projection is correct """
    self.assertIsInstance(self.model_result.combined_output_projection, nn.Linear)
    self.assertEqual(self.model_result.combined_output_projection.in_features, 3 * 17)
    self.assertEqual(self.model_result.combined_output_projection.out_features, 17)
    self.assertEqual(self.model_result.combined_output_projection.bias, None)


  @graded()
  def test_4(self):
    """1c-4-basic: Verify that self.target_vocab_projection is correct """
    self.assertIsInstance(self.model_result.target_vocab_projection, nn.Linear)
    self.assertEqual(self.model_result.target_vocab_projection.in_features, 17)
    self.assertEqual(self.model_result.target_vocab_projection.out_features, 5)
    self.assertEqual(self.model_result.target_vocab_projection.bias, None)

  @graded()
  def test_5(self):
    """1c-5-basic: Verify that self.dropout is correct """
    self.assertIsInstance(self.model_result.dropout, nn.Dropout)
    self.assertEqual(self.model_result.dropout.p, 0.34)

class Test_1d(GradedTestCase):
  def setUp(self):
    # Set Seeds
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(42)

    # Create Inputs
    input = setup()
    self.vocab = input[-1]

    # Initialize model
    self.model = run_model.NMT(
      embed_size = LARGE_EMBED_SIZE,
      hidden_size = LARGE_HIDDEN_SIZE,
      dropout_rate = NONZERO_DROPOUT_RATE,
      vocab = self.vocab
    )

    # Initialize soln model
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(42)
    self.soln_model = self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol).NMT(
      embed_size = LARGE_EMBED_SIZE,
      hidden_size = LARGE_HIDDEN_SIZE,
      dropout_rate = NONZERO_DROPOUT_RATE,
      vocab = self.vocab
    )

    self.source_lengths = [len(s) for s in input[0]]
    self.source_padded = self.soln_model.vocab.src.to_input_tensor(input[0], device=self.soln_model.device)
    self.enc_hidden, self.decode_hidden, self.decode_cell = test_encoding_hiddens(self.source_padded, self.source_lengths,
                                                                                  self.model, self.soln_model, self.vocab)

  @graded()
  def test_0(self):
    """1d-0-basic:  Sanity check for Encode.  Compares output to that of model with dummy data."""
    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    # Load training data & vocabulary
    train_data_src = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.es', 'src')
    train_data_tgt = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.en', 'tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    for src_sents, tgt_sents in run_model.batch_iter(train_data, batch_size=BATCH_SIZE, shuffle=True):
      src_sents = src_sents
      tgt_sents = tgt_sents
      break
    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = run_model.NMT(
      embed_size=EMBED_SIZE,
      hidden_size=HIDDEN_SIZE,
      dropout_rate=DROPOUT_RATE,
      vocab=vocab)
    # Configure for Testing
    reinitialize_layers(model)
    source_lengths = [len(s) for s in src_sents]
    source_padded = model.vocab.src.to_input_tensor(src_sents, device=model.device)

    # Load Outputs
    enc_hiddens_target = torch.load('./sanity_check_en_es_data/enc_hiddens.pkl')
    dec_init_state_target = torch.load('./sanity_check_en_es_data/dec_init_state.pkl')

    # Test
    with torch.no_grad():
        enc_hiddens_pred, dec_init_state_pred = model.encode(source_padded, source_lengths)
    self.assertTrue(np.allclose(enc_hiddens_target.numpy(),
                        enc_hiddens_pred.numpy())), "enc_hiddens is incorrect: it should be:\n {} but is:\n{}".format(
        enc_hiddens_target, enc_hiddens_pred)
    print("enc_hiddens Sanity Checks Passed!")
    self.assertTrue(np.allclose(dec_init_state_target[0].numpy(), dec_init_state_pred[
        0].numpy())), "dec_init_state[0] is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[0],
                                                                                              dec_init_state_pred[0])
    print("dec_init_state[0] Sanity Checks Passed!")
    self.assertTrue(np.allclose(dec_init_state_target[1].numpy(), dec_init_state_pred[
        1].numpy())), "dec_init_state[1] is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[1],
                                                                                              dec_init_state_pred[1])
    print("dec_init_state[1] Sanity Checks Passed!")

  @graded(is_hidden=True)
  def test_1(self):
      """1d-1-hidden: Encode Hiddens Check"""
      self.assertTrue(self.enc_hidden)

  @graded(is_hidden=True)
  def test_2(self):
      """1d-2-hidden: dec_state[0] Check"""
      self.assertTrue(self.decode_hidden)

  @graded(is_hidden=True)
  def test_3(self):
      """1d-3-hidden: dec_state[1] Check"""
      self.assertTrue(self.decode_cell)

class Test_1e(GradedTestCase):
  def setUp(self):
    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    # Load training data & vocabulary
    train_data_src = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.es', 'src')
    train_data_tgt = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.en', 'tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    for src_sents, tgt_sents in run_model.batch_iter(train_data, batch_size=BATCH_SIZE, shuffle=True):
      self.src_sents = src_sents
      self.tgt_sents = tgt_sents
      break
    self.vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    self.model = run_model.NMT(
      embed_size=EMBED_SIZE,
      hidden_size=HIDDEN_SIZE,
      dropout_rate=DROPOUT_RATE,
      vocab=self.vocab)

  @graded()
  def test_0(self):
    """1e-0-basic:  Sanity check for Decode.  Compares output to that of model with dummy data."""
    # Load Inputs
    dec_init_state = torch.load('./sanity_check_en_es_data/dec_init_state.pkl')
    enc_hiddens = torch.load('./sanity_check_en_es_data/enc_hiddens.pkl')
    enc_masks = torch.load('./sanity_check_en_es_data/enc_masks.pkl')
    target_padded = torch.load('./sanity_check_en_es_data/target_padded.pkl')

    # Load Outputs
    combined_outputs_target = torch.load('./sanity_check_en_es_data/combined_outputs.pkl')

    # Configure for Testing
    reinitialize_layers(self.model)
    COUNTER = [0]

    def stepFunction(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
      dec_state = torch.load('./sanity_check_en_es_data/step_dec_state_{}.pkl'.format(COUNTER[0]))
      o_t = torch.load('./sanity_check_en_es_data/step_o_t_{}.pkl'.format(COUNTER[0]))
      COUNTER[0] += 1
      return dec_state, o_t, None

    self.model.step = stepFunction

    # Run Tests
    with torch.no_grad():
      combined_outputs_pred = self.model.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
    self.assertTrue(np.allclose(combined_outputs_pred.numpy(),
                        combined_outputs_target.numpy())), "combined_outputs is should be:\n {}, but is:\n{}".format(
        combined_outputs_target, combined_outputs_pred)

  @graded(is_hidden=True)
  def test_1(self):
    """1e-1-hidden: Combined Outputs Check"""
    # Set Seeds
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create Inputs
    input = setup()
    self.vocab = input[-1]

    # Initialize model
    self.model = run_model.NMT(
        embed_size = LARGE_EMBED_SIZE,
        hidden_size = LARGE_HIDDEN_SIZE,
        dropout_rate = NONZERO_DROPOUT_RATE,
        vocab = self.vocab
    )

    # Initialize soln model
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    self.soln_model = self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol).NMT(
        embed_size = LARGE_EMBED_SIZE,
        hidden_size = LARGE_HIDDEN_SIZE,
        dropout_rate = NONZERO_DROPOUT_RATE,
        vocab = self.vocab
    )
    # To prevent dropout
    self.model.train(False)
    self.soln_model.train(False)

    self.source_lengths = [len(s) for s in input[0]]
    self.source_padded = self.soln_model.vocab.src.to_input_tensor(input[0], device=self.soln_model.device)
    self.target_padded = self.soln_model.vocab.tgt.to_input_tensor(input[1], device=self.soln_model.device)   # Tensor: (tgt_len, b)

    self.target = input[1]
    self.combined_outputs = test_combined_outputs(self.source_padded, self.source_lengths, self.target_padded,
                                                                                  self.model, self.soln_model, self.vocab)
    self.assertTrue(self.combined_outputs)

class Test_1f(GradedTestCase):
  def setUp(self):
    # Set Seeds
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create Inputs
    input = setup()
    self.vocab = input[-1]

    # Initialize model
    self.model = run_model.NMT(
        embed_size = LARGE_EMBED_SIZE,
        hidden_size = LARGE_HIDDEN_SIZE,
        dropout_rate = NONZERO_DROPOUT_RATE,
        vocab = self.vocab
    )

    # Initialize soln model
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    self.soln_model = self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol).NMT(
        embed_size = LARGE_EMBED_SIZE,
        hidden_size = LARGE_HIDDEN_SIZE,
        dropout_rate = NONZERO_DROPOUT_RATE,
        vocab = self.vocab
    )

    # To prevent dropout
    self.model.train(False)
    self.soln_model.train(False)
    # self.alt_soln_model.train(False)

    # Generate Inputs
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)

    Ybar_t = torch.randn(LARGE_BATCH_SIZE, LARGE_EMBED_SIZE + LARGE_HIDDEN_SIZE, dtype=torch.float)
    dec_init_state = (torch.randn(LARGE_BATCH_SIZE, LARGE_HIDDEN_SIZE, dtype=torch.float), torch.randn(LARGE_BATCH_SIZE, LARGE_HIDDEN_SIZE, dtype=torch.float))
    enc_hiddens = torch.randn(LARGE_BATCH_SIZE, 20, LARGE_HIDDEN_SIZE * 2, dtype=torch.float)
    enc_hiddens_proj = torch.randn(LARGE_BATCH_SIZE, 20, LARGE_HIDDEN_SIZE, dtype=torch.float)
    enc_masks = (torch.randn(LARGE_BATCH_SIZE, 20, dtype=torch.float) >= 0.5)

    self.dec_hidden_result, self.dec_state_result, self.o_t_result, self.e_t_result = \
        test_q1f(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks, self.model, self.soln_model)

  @graded()
  def test_0(self):
    """1f-0-basic:  Sanity check for Step.  Compares output to that of model with dummy data."""
    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    # Load training data & vocabulary
    train_data_src = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.es', 'src')
    train_data_tgt = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.en', 'tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    for src_sents, tgt_sents in run_model.batch_iter(train_data, batch_size=BATCH_SIZE, shuffle=True):
      self.src_sents = src_sents
      self.tgt_sents = tgt_sents
      break
    self.vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    self.model = run_model.NMT(
      embed_size=EMBED_SIZE,
      hidden_size=HIDDEN_SIZE,
      dropout_rate=DROPOUT_RATE,
      vocab=self.vocab)

    reinitialize_layers(self.model)
    # Inputs
    Ybar_t = torch.load('./sanity_check_en_es_data/Ybar_t.pkl')
    dec_init_state = torch.load('./sanity_check_en_es_data/dec_init_state.pkl')
    enc_hiddens = torch.load('./sanity_check_en_es_data/enc_hiddens.pkl')
    enc_masks = torch.load('./sanity_check_en_es_data/enc_masks.pkl')
    enc_hiddens_proj = torch.load('./sanity_check_en_es_data/enc_hiddens_proj.pkl')

    # Output
    dec_state_target = torch.load('./sanity_check_en_es_data/dec_state.pkl')
    o_t_target = torch.load('./sanity_check_en_es_data/o_t.pkl')
    e_t_target = torch.load('./sanity_check_en_es_data/e_t.pkl')

    # Run Tests
    with torch.no_grad():
        dec_state_pred, o_t_pred, e_t_pred = self.model.step(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj,
                                                        enc_masks)
    self.assertTrue(np.allclose(dec_state_target[0].numpy(), dec_state_pred[
        0].numpy()), "decoder_state[0] should be:\n {} but is:\n{}".format(dec_state_target[0],
                                                                                             dec_state_pred[0]))
    print("dec_state[0] Sanity Checks Passed!")
    self.assertTrue(np.allclose(dec_state_target[1].numpy(), dec_state_pred[
        1].numpy()), "decoder_state[1] should be:\n {} but is:\n{}".format(dec_state_target[1],
                                                                                             dec_state_pred[1]))
    print("dec_state[1] Sanity Checks Passed!")
    self.assertTrue(np.allclose(o_t_target.numpy(),
                        o_t_pred.numpy()), "combined_output should be:\n {} but is:\n{}".format(
        o_t_target, o_t_pred))
    print("combined_output  Sanity Checks Passed!")
    self.assertTrue(
        np.allclose(e_t_target.numpy(), e_t_pred.numpy()), "e_t should be:\n {} but is:\n{}".format(
        e_t_target, e_t_pred))

  @graded(is_hidden=True)
  def test_1(self):
    """1f-1-hidden: Decoder Hiddens Check"""
    self.assertTrue(self.dec_hidden_result)

  @graded(is_hidden=True)
  def test_2(self):
    """1f-2-hidden: Decoder State Check"""
    self.assertTrue(self.dec_state_result)

  @graded(is_hidden=True)
  def test_3(self):
    """1f-3-hidden: o_t Check"""
    self.assertTrue(self.o_t_result)

  @graded(is_hidden=True)
  def test_4(self):
    """1f-4-hidden: e_t Check"""
    self.assertTrue(self.e_t_result)

class Test_1g(GradedTestCase):
    @graded(is_hidden=True)
    def test_0(self):
        """1g-0-hidden: BLEU score check"""
        args = {
            'TEST_OUTPUT_FILE': './run_model/test_outputs.txt',
            'TEST_GOLD_FILE': './chr_en_data/test.en'
        }
        self.assertTrue(os.path.exists(args['TEST_OUTPUT_FILE']),
                        f'Output test file ({args["TEST_OUTPUT_FILE"]}) does not exist. To generate this file, follow these steps:\n'
                        '1. Generate vocab.py (sh run.sh vocab)\n'
                        '2. Generate and train a model (sh run.sh train)\n'
                        '3. Test trained model (takes 30min - 1 hour to train) (sh run.sh test)')
        self.assertGreater(bleu(args), 10, "Must achieve a BLEU score greater than 10.")

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='tester.py'))
  CourseTestRunner().run(assignment)