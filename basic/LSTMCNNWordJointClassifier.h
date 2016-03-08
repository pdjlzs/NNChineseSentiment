/*
 * LSTMCNNWordClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_LSTMCNNWordClassifier_H_
#define SRC_LSTMCNNWordClassifier_H_

#include <iostream>

#include <assert.h>
#include "ExampleMultiSeg.h"
#include "Feature.h"

#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class LSTMCNNWordClassifier {
public:
  LSTMCNNWordClassifier() {
    _dropOut = 0.5;
  }
  ~LSTMCNNWordClassifier() {

  }

public:

  vector<LookupTable<xpu> > _seg_words;

  int _wordcontext;
  int _wordSize;
  int _wordDim;
  bool _b_wordEmb_finetune;
  int _wordHiddenSize;
  int _word_cnn_iSize;
  int _token_representation_size;

  int _hiddenSize;

  UniLayer<xpu> _tanh_project;
  UniLayer<xpu> _olayer_linear;

  vector<LSTM_STD<xpu> > _rnn_left;
  vector<LSTM_STD<xpu> > _rnn_right;
  vector<UniLayer<xpu> > _cnn_project;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3 min

  int _poolmanners;

  Alphabet _segStylelabelAlphabet;

public:

  inline void init(Alphabet segStylAlpbt, const NRMat<dtype>& ctbEmb, const NRMat<dtype>& pkuEmb, const NRMat<dtype>& msrEmb, const NRMat<dtype>& charEmb,
      int wordcontext, int labelSize, int wordHiddenSize, int hiddenSize, int nbest) {
    _wordcontext = wordcontext;
    _wordSize = ctbEmb.nrows();
    _wordDim = ctbEmb.ncols();
    _poolmanners = 3;

    _labelSize = labelSize;
    _hiddenSize = hiddenSize;
    _wordHiddenSize = wordHiddenSize;
    _token_representation_size = _wordDim;

    _word_cnn_iSize = _token_representation_size * (2 * _wordcontext + 1);

    _segStylelabelAlphabet = segStylAlpbt;
    int seg_style = _segStylelabelAlphabet.size();
    _rnn_left.resize(seg_style);
    _rnn_right.resize(seg_style);
    _cnn_project.resize(seg_style);
    for (int i = 0; i < seg_style; i++) {
      _rnn_left[i].initial(_wordHiddenSize, _word_cnn_iSize, true, 40 * i);
      _rnn_right[i].initial(_wordHiddenSize, _word_cnn_iSize, false, 70 * i);
      _cnn_project[i].initial(_wordHiddenSize, 2 * _wordHiddenSize, true, 90 * i, 0);
      LookupTable<xpu> words;
      if (segStylAlpbt.from_id(i) == "ctb")
        words.initial(ctbEmb);
      if (segStylAlpbt.from_id(i) == "pku")
        words.initial(pkuEmb);
      if (segStylAlpbt.from_id(i) == "msr")
        words.initial(msrEmb);
      if (segStylAlpbt.from_id(i) == "cha")
        words.initial(charEmb);
      _seg_words.push_back(words);

    }

    _tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize * _segStylelabelAlphabet.size() * nbest, true, 50, 0);
    _olayer_linear.initial(_labelSize, hiddenSize, false, 60, 2);

    _eval.reset();

    _remove = 0;

  }

  inline void release() {

    for (int i = 0; i < _segStylelabelAlphabet.size(); i++) {
      _seg_words[i].release();
      _rnn_left[i].release();
      _rnn_right[i].release();
      _cnn_project[i].release();
    }

    _tanh_project.release();
    _olayer_linear.release();
  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      vector<vector<Tensor<xpu, 3, dtype> > > wordprime, wordprimeLoss, wordprimeMask;
      vector<vector<Tensor<xpu, 3, dtype> > > wordrepresent, wordrepresentLoss;
      vector<vector<Tensor<xpu, 3, dtype> > > input, inputLoss;

      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_i;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_o;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_f;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_mc;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_c;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_my;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left, rnn_hidden_leftLoss;

      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_i;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_o;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_f;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_mc;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_c;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_my;
      vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right, rnn_hidden_rightLoss;

      vector<vector<Tensor<xpu, 3, dtype> > > midhidden, midhiddenLoss;

      vector<vector<Tensor<xpu, 3, dtype> > > hidden, hiddenLoss;
      vector<vector<vector<Tensor<xpu, 2, dtype> > > > pool, poolLoss;
      vector<vector<vector<Tensor<xpu, 3, dtype> > > > poolIndex;

      vector<vector<Tensor<xpu, 2, dtype> > > poolmerge, poolmergeLoss;
      vector<Tensor<xpu, 2, dtype> > nbestmerge, nbestmergeLoss;
      Tensor<xpu, 2, dtype> sentmerge, sentmergeLoss;
      Tensor<xpu, 2, dtype> project, projectLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      int seg_style = example.m_features.size();

      // vector resize
      wordprime.resize(seg_style);
      wordprimeLoss.resize(seg_style);
      wordprimeMask.resize(seg_style);
      wordrepresent.resize(seg_style);
      wordrepresentLoss.resize(seg_style);
      input.resize(seg_style);
      inputLoss.resize(seg_style);

      rnn_hidden_left_i.resize(seg_style);
      rnn_hidden_left_o.resize(seg_style);
      rnn_hidden_left_f.resize(seg_style);
      rnn_hidden_left_mc.resize(seg_style);
      rnn_hidden_left_c.resize(seg_style);
      rnn_hidden_left_my.resize(seg_style);
      rnn_hidden_left.resize(seg_style);
      rnn_hidden_leftLoss.resize(seg_style);

      rnn_hidden_right_i.resize(seg_style);
      rnn_hidden_right_o.resize(seg_style);
      rnn_hidden_right_f.resize(seg_style);
      rnn_hidden_right_mc.resize(seg_style);
      rnn_hidden_right_c.resize(seg_style);
      rnn_hidden_right_my.resize(seg_style);
      rnn_hidden_right.resize(seg_style);
      rnn_hidden_rightLoss.resize(seg_style);

      midhidden.resize(seg_style);
      midhiddenLoss.resize(seg_style);

      hidden.resize(seg_style);
      hiddenLoss.resize(seg_style);
      pool.resize(seg_style);
      poolLoss.resize(seg_style);
      poolIndex.resize(seg_style);

      poolmerge.resize(seg_style);
      poolmergeLoss.resize(seg_style);

      nbestmerge.resize(seg_style);
      nbestmergeLoss.resize(seg_style);

      int total_sent_num = 0;
      for (int i = 0; i < seg_style; i++) {
        total_sent_num += 1;
        int nbest_num = example.m_features[i].size();

        wordprime[i].resize(nbest_num);
        wordprimeLoss[i].resize(nbest_num);
        wordprimeMask[i].resize(nbest_num);
        wordrepresent[i].resize(nbest_num);
        wordrepresentLoss[i].resize(nbest_num);
        input[i].resize(nbest_num);
        inputLoss[i].resize(nbest_num);

        rnn_hidden_left_i[i].resize(nbest_num);
        rnn_hidden_left_o[i].resize(nbest_num);
        rnn_hidden_left_f[i].resize(nbest_num);
        rnn_hidden_left_mc[i].resize(nbest_num);
        rnn_hidden_left_c[i].resize(nbest_num);
        rnn_hidden_left_my[i].resize(nbest_num);
        rnn_hidden_left[i].resize(nbest_num);
        rnn_hidden_leftLoss[i].resize(nbest_num);

        rnn_hidden_right_i[i].resize(nbest_num);
        rnn_hidden_right_o[i].resize(nbest_num);
        rnn_hidden_right_f[i].resize(nbest_num);
        rnn_hidden_right_mc[i].resize(nbest_num);
        rnn_hidden_right_c[i].resize(nbest_num);
        rnn_hidden_right_my[i].resize(nbest_num);
        rnn_hidden_right[i].resize(nbest_num);
        rnn_hidden_rightLoss[i].resize(nbest_num);

        midhidden[i].resize(nbest_num);
        midhiddenLoss[i].resize(nbest_num);

        hidden[i].resize(nbest_num);
        hiddenLoss[i].resize(nbest_num);
        pool[i].resize(nbest_num);
        poolLoss[i].resize(nbest_num);
        poolIndex[i].resize(nbest_num);

        poolmerge[i].resize(nbest_num);
        poolmergeLoss[i].resize(nbest_num);

        for (int j = 0; j < nbest_num; j++) {
          pool[i][j].resize(_poolmanners);
          poolLoss[i][j].resize(_poolmanners);
          poolIndex[i][j].resize(_poolmanners);
        }
      }
      //initialize
      //int idx = seq_size - 1;

      for (int i = 0; i < seg_style; i++) {

        int nbest_num = example.m_features[i].size();
        for (int j = 0; j < nbest_num; j++) {

          const Feature& feature = example.m_features[i][j];
          int word_num = feature.words.size();
          int word_cnn_iSize = _word_cnn_iSize;
          int wordHiddenSize = _wordHiddenSize;

          wordprime[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
          wordprimeLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
          wordprimeMask[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_one);
          wordrepresent[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
          wordrepresentLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);

          input[i][j] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);
          inputLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

          rnn_hidden_left_i[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_left_o[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_left_f[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_left_mc[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_left_c[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_left_my[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_left[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_leftLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

          rnn_hidden_right_i[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_right_o[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_right_f[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_right_mc[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_right_c[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_right_my[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_right[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_rightLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

          midhidden[i][j] = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);
          midhiddenLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);

          hidden[i][j] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
          hiddenLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

          for (int idm = 0; idm < _poolmanners; idm++) {
            pool[i][j][idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
            poolLoss[i][j][idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
            poolIndex[i][j][idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
          }

          poolmerge[i][j] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
          poolmergeLoss[i][j] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);

        }

        nbestmerge[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * nbest_num), d_zero);
        nbestmergeLoss[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * nbest_num), d_zero);
      }

      sentmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * total_sent_num), d_zero);
      sentmergeLoss = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * total_sent_num), d_zero);

      project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      projectLoss = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      //forward propagation
      //input setting, and linear setting
      for (int i = 0; i < seg_style; i++) {
        for (int j = 0; j < example.m_features[i].size(); j++) {

          const Feature& feature = example.m_features[i][j];
          int curcontext = _wordcontext;

          const vector<int>& words = feature.words;
          int word_num = words.size();
          //linear features should not be dropped out

          srand(iter * example_num + count * seg_style + i + j);

          for (int idy = 0; idy < word_num; idy++) {
            _seg_words[i].GetEmb(words[idy], wordprime[i][j][idy]);
          }

          //word dropout
          for (int idy = 0; idy < word_num; idy++) {
            dropoutcol(wordprimeMask[i][j][idy], _dropOut);
            wordprime[i][j][idy] = wordprime[i][j][idy] * wordprimeMask[i][j][idy];
          }

          //word representation
          for (int idy = 0; idy < word_num; idy++) {
            wordrepresent[i][j][idy] += wordprime[i][j][idy];
          }

          windowlized(wordrepresent[i][j], input[i][j], curcontext);

          _rnn_left[i].ComputeForwardScore(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
              rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j]);
          _rnn_right[i].ComputeForwardScore(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j],
              rnn_hidden_right_mc[i][j], rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j]);

          for (int idy = 0; idy < word_num; idy++) {
            concat(rnn_hidden_left[i][j][idy], rnn_hidden_right[i][j][idy], midhidden[i][j][idy]);
          }

          _cnn_project[i].ComputeForwardScore(midhidden[i][j], hidden[i][j]);

          //word pooling
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            avgpool_forward(hidden[i][j], pool[i][j][0], poolIndex[i][j][0]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            maxpool_forward(hidden[i][j], pool[i][j][1], poolIndex[i][j][1]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            minpool_forward(hidden[i][j], pool[i][j][2], poolIndex[i][j][2]);
          }

        }
      }

      // sentence
      for (int i = 0; i < seg_style; i++) {
        for (int j = 0; j < example.m_features[i].size(); j++) {
          concat(pool[i][j], poolmerge[i][j]);
        }
        concat(poolmerge[i], nbestmerge[i]);
      }
      concat(nbestmerge, sentmerge);

      _tanh_project.ComputeForwardScore(sentmerge, project);
      _olayer_linear.ComputeForwardScore(project, output);

      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      //sentence
      _olayer_linear.ComputeBackwardLoss(project, output, outputLoss, projectLoss);
      _tanh_project.ComputeBackwardLoss(sentmerge, project, projectLoss, sentmergeLoss);

      unconcat(nbestmergeLoss, sentmergeLoss);
      for (int i = 0; i < seg_style; i++) {
        unconcat(poolmergeLoss[i], nbestmergeLoss[i]);
        for (int j = 0; j < example.m_features[i].size(); j++) {
          unconcat(poolLoss[i][j], poolmergeLoss[i][j]);
        }
      }

      for (int i = 0; i < seg_style; i++) {
        int nbest_num = example.m_features[i].size();
        for (int j = 0; j < nbest_num; j++) {

          const Feature& feature = example.m_features[i][j];
          int curcontext = _wordcontext;
          const vector<int>& words = feature.words;
          int word_num = words.size();

          //word pooling
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            pool_backward(poolLoss[i][j][0], poolIndex[i][j][0], hiddenLoss[i][j]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            pool_backward(poolLoss[i][j][1], poolIndex[i][j][1], hiddenLoss[i][j]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            pool_backward(poolLoss[i][j][2], poolIndex[i][j][2], hiddenLoss[i][j]);
          }

          _cnn_project[i].ComputeBackwardLoss(midhidden[i][j], hidden[i][j], hiddenLoss[i][j], midhiddenLoss[i][j]);

          for (int idy = 0; idy < word_num; idy++) {
            unconcat(rnn_hidden_leftLoss[i][j][idy], rnn_hidden_rightLoss[i][j][idy], midhiddenLoss[i][j][idy]);
          }

          _rnn_left[i].ComputeBackwardLoss(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
              rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j], rnn_hidden_leftLoss[i][j], inputLoss[i][j]);
          _rnn_right[i].ComputeBackwardLoss(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j],
              rnn_hidden_right_mc[i][j], rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j], rnn_hidden_rightLoss[i][j],
              inputLoss[i][j]);

          windowlized_backward(wordrepresentLoss[i][j], inputLoss[i][j], curcontext);

          //word representation
          for (int idy = 0; idy < word_num; idy++) {
            wordprimeLoss[i][j][idy] += wordrepresentLoss[i][j][idy];
          }

          if (_seg_words[i].bEmbFineTune()) {
            for (int idy = 0; idy < word_num; idy++) {
              wordprimeLoss[i][j][idy] = wordprimeLoss[i][j][idy] * wordprimeMask[i][j][idy];
              _seg_words[i].EmbLoss(words[idy], wordprimeLoss[i][j][idy]);
            }
          }

        }
      }

      //release

      for (int i = 0; i < seg_style; i++) {
        int nbest_num = example.m_features[i].size();
        for (int j = 0; j < nbest_num; j++) {

          FreeSpace(&(wordprime[i][j]));
          FreeSpace(&(wordprimeLoss[i][j]));
          FreeSpace(&(wordprimeMask[i][j]));
          FreeSpace(&(wordrepresent[i][j]));
          FreeSpace(&(wordrepresentLoss[i][j]));
          FreeSpace(&(input[i][j]));
          FreeSpace(&(inputLoss[i][j]));
          FreeSpace(&(hidden[i][j]));
          FreeSpace(&(hiddenLoss[i][j]));
          FreeSpace(&(rnn_hidden_left_i[i][j]));
          FreeSpace(&(rnn_hidden_left_o[i][j]));
          FreeSpace(&(rnn_hidden_left_f[i][j]));
          FreeSpace(&(rnn_hidden_left_mc[i][j]));
          FreeSpace(&(rnn_hidden_left_c[i][j]));
          FreeSpace(&(rnn_hidden_left_my[i][j]));
          FreeSpace(&(rnn_hidden_left[i][j]));
          FreeSpace(&(rnn_hidden_leftLoss[i][j]));

          FreeSpace(&(rnn_hidden_right_i[i][j]));
          FreeSpace(&(rnn_hidden_right_o[i][j]));
          FreeSpace(&(rnn_hidden_right_f[i][j]));
          FreeSpace(&(rnn_hidden_right_mc[i][j]));
          FreeSpace(&(rnn_hidden_right_c[i][j]));
          FreeSpace(&(rnn_hidden_right_my[i][j]));
          FreeSpace(&(rnn_hidden_right[i][j]));
          FreeSpace(&(rnn_hidden_rightLoss[i][j]));

          FreeSpace(&(midhidden[i][j]));
          FreeSpace(&(midhiddenLoss[i][j]));

          FreeSpace(&(poolmerge[i][j]));
          FreeSpace(&(poolmergeLoss[i][j]));
          for (int idm = 0; idm < _poolmanners; idm++) {
            FreeSpace(&(pool[i][j][idm]));
            FreeSpace(&(poolLoss[i][j][idm]));
            FreeSpace(&(poolIndex[i][j][idm]));
          }
        }
        FreeSpace(&(nbestmerge[i]));
        FreeSpace(&(nbestmergeLoss[i]));
      }
      FreeSpace(&sentmerge);
      FreeSpace(&sentmergeLoss);
      FreeSpace(&project);
      FreeSpace(&projectLoss);
      FreeSpace(&output);
      FreeSpace(&outputLoss);
    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const vector<int>& linears, const vector<vector<Feature> >& features, vector<dtype>& results) {

    vector<vector<Tensor<xpu, 3, dtype> > > wordprime;
    vector<vector<Tensor<xpu, 3, dtype> > > wordrepresent;
    vector<vector<Tensor<xpu, 3, dtype> > > input;

    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_i;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_o;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_f;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_mc;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_c;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_my;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left;

    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_i;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_o;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_f;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_mc;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_c;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_my;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right;

    vector<vector<Tensor<xpu, 3, dtype> > > midhidden;

    vector<vector<Tensor<xpu, 3, dtype> > > hidden;
    vector<vector<vector<Tensor<xpu, 2, dtype> > > > pool;
    vector<vector<vector<Tensor<xpu, 3, dtype> > > > poolIndex;

    vector<vector<Tensor<xpu, 2, dtype> > > poolmerge;
    vector<Tensor<xpu, 2, dtype> > nbestmerge;
    Tensor<xpu, 2, dtype> sentmerge;
    Tensor<xpu, 2, dtype> project;
    Tensor<xpu, 2, dtype> output;

    int seg_style = features.size();

    // vector resize
    wordprime.resize(seg_style);
    wordrepresent.resize(seg_style);
    input.resize(seg_style);

    rnn_hidden_left_i.resize(seg_style);
    rnn_hidden_left_o.resize(seg_style);
    rnn_hidden_left_f.resize(seg_style);
    rnn_hidden_left_mc.resize(seg_style);
    rnn_hidden_left_c.resize(seg_style);
    rnn_hidden_left_my.resize(seg_style);
    rnn_hidden_left.resize(seg_style);

    rnn_hidden_right_i.resize(seg_style);
    rnn_hidden_right_o.resize(seg_style);
    rnn_hidden_right_f.resize(seg_style);
    rnn_hidden_right_mc.resize(seg_style);
    rnn_hidden_right_c.resize(seg_style);
    rnn_hidden_right_my.resize(seg_style);
    rnn_hidden_right.resize(seg_style);

    midhidden.resize(seg_style);

    hidden.resize(seg_style);
    pool.resize(seg_style);
    poolIndex.resize(seg_style);

    poolmerge.resize(seg_style);

    nbestmerge.resize(seg_style);

    int total_sent_num = 0;
    for (int i = 0; i < seg_style; i++) {
      total_sent_num += 1;
      int nbest_num = features[i].size();

      wordprime[i].resize(nbest_num);
      wordrepresent[i].resize(nbest_num);
      input[i].resize(nbest_num);

      rnn_hidden_left_i[i].resize(nbest_num);
      rnn_hidden_left_o[i].resize(nbest_num);
      rnn_hidden_left_f[i].resize(nbest_num);
      rnn_hidden_left_mc[i].resize(nbest_num);
      rnn_hidden_left_c[i].resize(nbest_num);
      rnn_hidden_left_my[i].resize(nbest_num);
      rnn_hidden_left[i].resize(nbest_num);

      rnn_hidden_right_i[i].resize(nbest_num);
      rnn_hidden_right_o[i].resize(nbest_num);
      rnn_hidden_right_f[i].resize(nbest_num);
      rnn_hidden_right_mc[i].resize(nbest_num);
      rnn_hidden_right_c[i].resize(nbest_num);
      rnn_hidden_right_my[i].resize(nbest_num);
      rnn_hidden_right[i].resize(nbest_num);

      midhidden[i].resize(nbest_num);

      hidden[i].resize(nbest_num);
      pool[i].resize(nbest_num);
      poolIndex[i].resize(nbest_num);

      poolmerge[i].resize(nbest_num);

      for (int j = 0; j < nbest_num; j++) {
        pool[i][j].resize(_poolmanners);
        poolIndex[i][j].resize(_poolmanners);
      }
    }
    //initialize
    for (int i = 0; i < seg_style; i++) {

      int nbest_num = features[i].size();
      for (int j = 0; j < nbest_num; j++) {
        int word_cnn_iSize = _word_cnn_iSize;
        int wordHiddenSize = _wordHiddenSize;
        const Feature& feature = features[i][j];
        int word_num = feature.words.size();

        wordprime[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
        wordrepresent[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);

        input[i][j] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

        rnn_hidden_left_i[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_o[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_f[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_mc[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_c[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_my[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

        rnn_hidden_right_i[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_o[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_f[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_mc[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_c[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_my[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

        midhidden[i][j] = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);

        hidden[i][j] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

        for (int idm = 0; idm < _poolmanners; idm++) {
          pool[i][j][idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
          poolIndex[i][j][idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
        }

        poolmerge[i][j] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);

      }

      nbestmerge[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * nbest_num), d_zero);
    }

    sentmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * total_sent_num), d_zero);

    project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
    //forward propagation
    //input setting, and linear setting
    for (int i = 0; i < seg_style; i++) {
      for (int j = 0; j < features[i].size(); j++) {

        const Feature& feature = features[i][j];
        int curcontext = _wordcontext;

        const vector<int>& words = feature.words;
        int word_num = words.size();
        //linear features should not be dropped out

        for (int idy = 0; idy < word_num; idy++) {
          _seg_words[i].GetEmb(words[idy], wordprime[i][j][idy]);
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          wordrepresent[i][j][idy] += wordprime[i][j][idy];
        }

        windowlized(wordrepresent[i][j], input[i][j], curcontext);

        _rnn_left[i].ComputeForwardScore(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
            rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j]);
        _rnn_right[i].ComputeForwardScore(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j], rnn_hidden_right_mc[i][j],
            rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j]);

        for (int idy = 0; idy < word_num; idy++) {
          concat(rnn_hidden_left[i][j][idy], rnn_hidden_right[i][j][idy], midhidden[i][j][idy]);
        }

        _cnn_project[i].ComputeForwardScore(midhidden[i][j], hidden[i][j]);

        //word pooling
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(hidden[i][j], pool[i][j][0], poolIndex[i][j][0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(hidden[i][j], pool[i][j][1], poolIndex[i][j][1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(hidden[i][j], pool[i][j][2], poolIndex[i][j][2]);
        }
      }
    }

    // sentence
    for (int i = 0; i < seg_style; i++) {
      for (int j = 0; j < features[i].size(); j++) {
        concat(pool[i][j], poolmerge[i][j]);
      }
      concat(poolmerge[i], nbestmerge[i]);
    }
    concat(nbestmerge, sentmerge);

    _tanh_project.ComputeForwardScore(sentmerge, project);
    _olayer_linear.ComputeForwardScore(project, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    for (int i = 0; i < seg_style; i++) {
      int nbest_num = features[i].size();
      for (int j = 0; j < nbest_num; j++) {

        FreeSpace(&(wordprime[i][j]));
        FreeSpace(&(wordrepresent[i][j]));
        FreeSpace(&(input[i][j]));
        FreeSpace(&(hidden[i][j]));
        FreeSpace(&(rnn_hidden_left_i[i][j]));
        FreeSpace(&(rnn_hidden_left_o[i][j]));
        FreeSpace(&(rnn_hidden_left_f[i][j]));
        FreeSpace(&(rnn_hidden_left_mc[i][j]));
        FreeSpace(&(rnn_hidden_left_c[i][j]));
        FreeSpace(&(rnn_hidden_left_my[i][j]));
        FreeSpace(&(rnn_hidden_left[i][j]));

        FreeSpace(&(rnn_hidden_right_i[i][j]));
        FreeSpace(&(rnn_hidden_right_o[i][j]));
        FreeSpace(&(rnn_hidden_right_f[i][j]));
        FreeSpace(&(rnn_hidden_right_mc[i][j]));
        FreeSpace(&(rnn_hidden_right_c[i][j]));
        FreeSpace(&(rnn_hidden_right_my[i][j]));
        FreeSpace(&(rnn_hidden_right[i][j]));

        FreeSpace(&(midhidden[i][j]));

        FreeSpace(&(poolmerge[i][j]));
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(pool[i][j][idm]));
          FreeSpace(&(poolIndex[i][j][idm]));
        }
      }
      FreeSpace(&(nbestmerge[i]));
    }
    FreeSpace(&sentmerge);
    FreeSpace(&project);
    FreeSpace(&output);

    return optLabel;
  }

  dtype computeScore(const Example& example) {

    vector<vector<Tensor<xpu, 3, dtype> > > wordprime;
    vector<vector<Tensor<xpu, 3, dtype> > > wordrepresent;
    vector<vector<Tensor<xpu, 3, dtype> > > input;

    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_i;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_o;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_f;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_mc;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_c;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left_my;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_left;

    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_i;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_o;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_f;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_mc;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_c;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right_my;
    vector<vector<Tensor<xpu, 3, dtype> > > rnn_hidden_right;

    vector<vector<Tensor<xpu, 3, dtype> > > midhidden;

    vector<vector<Tensor<xpu, 3, dtype> > > hidden;
    vector<vector<vector<Tensor<xpu, 2, dtype> > > > pool;
    vector<vector<vector<Tensor<xpu, 3, dtype> > > > poolIndex;

    vector<vector<Tensor<xpu, 2, dtype> > > poolmerge;
    vector<Tensor<xpu, 2, dtype> > nbestmerge;
    Tensor<xpu, 2, dtype> sentmerge;
    Tensor<xpu, 2, dtype> project;
    Tensor<xpu, 2, dtype> output;

    int seg_style = example.m_features.size();

    // vector resize
    wordprime.resize(seg_style);
    wordrepresent.resize(seg_style);
    input.resize(seg_style);

    rnn_hidden_left_i.resize(seg_style);
    rnn_hidden_left_o.resize(seg_style);
    rnn_hidden_left_f.resize(seg_style);
    rnn_hidden_left_mc.resize(seg_style);
    rnn_hidden_left_c.resize(seg_style);
    rnn_hidden_left_my.resize(seg_style);
    rnn_hidden_left.resize(seg_style);

    rnn_hidden_right_i.resize(seg_style);
    rnn_hidden_right_o.resize(seg_style);
    rnn_hidden_right_f.resize(seg_style);
    rnn_hidden_right_mc.resize(seg_style);
    rnn_hidden_right_c.resize(seg_style);
    rnn_hidden_right_my.resize(seg_style);
    rnn_hidden_right.resize(seg_style);

    midhidden.resize(seg_style);

    hidden.resize(seg_style);
    pool.resize(seg_style);
    poolIndex.resize(seg_style);

    poolmerge.resize(seg_style);

    nbestmerge.resize(seg_style);

    int total_sent_num = 0;
    for (int i = 0; i < seg_style; i++) {
      total_sent_num += 1;
      int nbest_num = example.m_features[i].size();

      wordprime[i].resize(nbest_num);
      wordrepresent[i].resize(nbest_num);
      input[i].resize(nbest_num);

      rnn_hidden_left_i[i].resize(nbest_num);
      rnn_hidden_left_o[i].resize(nbest_num);
      rnn_hidden_left_f[i].resize(nbest_num);
      rnn_hidden_left_mc[i].resize(nbest_num);
      rnn_hidden_left_c[i].resize(nbest_num);
      rnn_hidden_left_my[i].resize(nbest_num);
      rnn_hidden_left[i].resize(nbest_num);

      rnn_hidden_right_i[i].resize(nbest_num);
      rnn_hidden_right_o[i].resize(nbest_num);
      rnn_hidden_right_f[i].resize(nbest_num);
      rnn_hidden_right_mc[i].resize(nbest_num);
      rnn_hidden_right_c[i].resize(nbest_num);
      rnn_hidden_right_my[i].resize(nbest_num);
      rnn_hidden_right[i].resize(nbest_num);

      midhidden[i].resize(nbest_num);

      hidden[i].resize(nbest_num);
      pool[i].resize(nbest_num);
      poolIndex[i].resize(nbest_num);

      poolmerge[i].resize(nbest_num);

      for (int j = 0; j < nbest_num; j++) {
        pool[i][j].resize(_poolmanners);
        poolIndex[i][j].resize(_poolmanners);
      }
    }
    //initialize
    for (int i = 0; i < seg_style; i++) {

      int nbest_num = example.m_features[i].size();
      for (int j = 0; j < nbest_num; j++) {
        int word_cnn_iSize = _word_cnn_iSize;
        int wordHiddenSize = _wordHiddenSize;
        const Feature& feature = example.m_features[i][j];
        int word_num = feature.words.size();

        wordprime[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
        wordrepresent[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);

        input[i][j] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

        rnn_hidden_left_i[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_o[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_f[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_mc[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_c[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left_my[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_left[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

        rnn_hidden_right_i[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_o[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_f[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_mc[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_c[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right_my[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

        midhidden[i][j] = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);

        hidden[i][j] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

        for (int idm = 0; idm < _poolmanners; idm++) {
          pool[i][j][idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
          poolIndex[i][j][idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
        }

        poolmerge[i][j] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);

      }

      nbestmerge[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * nbest_num), d_zero);
    }

    sentmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize * total_sent_num), d_zero);

    project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

    //forward propagation
    //input setting, and linear setting
    for (int i = 0; i < seg_style; i++) {
      for (int j = 0; j < example.m_features[i].size(); j++) {

        const Feature& feature = example.m_features[i][j];
        int curcontext = _wordcontext;

        const vector<int>& words = feature.words;
        int word_num = words.size();
        //linear features should not be dropped out

        for (int idy = 0; idy < word_num; idy++) {
          _seg_words[i].GetEmb(words[idy], wordprime[i][j][idy]);
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          wordrepresent[i][j][idy] += wordprime[i][j][idy];
        }

        windowlized(wordrepresent[i][j], input[i][j], curcontext);

        _rnn_left[i].ComputeForwardScore(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
            rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j]);
        _rnn_right[i].ComputeForwardScore(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j], rnn_hidden_right_mc[i][j],
            rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j]);

        for (int idy = 0; idy < word_num; idy++) {
          concat(rnn_hidden_left[i][j][idy], rnn_hidden_right[i][j][idy], midhidden[i][j][idy]);
        }

        _cnn_project[i].ComputeForwardScore(midhidden[i][j], hidden[i][j]);

        //word pooling
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(hidden[i][j], pool[i][j][0], poolIndex[i][j][0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(hidden[i][j], pool[i][j][1], poolIndex[i][j][1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(hidden[i][j], pool[i][j][2], poolIndex[i][j][2]);
        }
      }
    }

    // sentence
    for (int i = 0; i < seg_style; i++) {
      for (int j = 0; j < example.m_features[i].size(); j++) {
        concat(pool[i][j], poolmerge[i][j]);
      }
      concat(poolmerge[i], nbestmerge[i]);
    }
    concat(nbestmerge, sentmerge);

    _tanh_project.ComputeForwardScore(sentmerge, project);
    _olayer_linear.ComputeForwardScore(project, output);

    dtype cost = softmax_cost(output, example.m_labels);

    //release
    for (int i = 0; i < seg_style; i++) {
      int nbest_num = example.m_features[i].size();
      for (int j = 0; j < nbest_num; j++) {

        FreeSpace(&(wordprime[i][j]));
        FreeSpace(&(wordrepresent[i][j]));
        FreeSpace(&(input[i][j]));
        FreeSpace(&(hidden[i][j]));
        FreeSpace(&(rnn_hidden_left_i[i][j]));
        FreeSpace(&(rnn_hidden_left_o[i][j]));
        FreeSpace(&(rnn_hidden_left_f[i][j]));
        FreeSpace(&(rnn_hidden_left_mc[i][j]));
        FreeSpace(&(rnn_hidden_left_c[i][j]));
        FreeSpace(&(rnn_hidden_left_my[i][j]));
        FreeSpace(&(rnn_hidden_left[i][j]));

        FreeSpace(&(rnn_hidden_right_i[i][j]));
        FreeSpace(&(rnn_hidden_right_o[i][j]));
        FreeSpace(&(rnn_hidden_right_f[i][j]));
        FreeSpace(&(rnn_hidden_right_mc[i][j]));
        FreeSpace(&(rnn_hidden_right_c[i][j]));
        FreeSpace(&(rnn_hidden_right_my[i][j]));
        FreeSpace(&(rnn_hidden_right[i][j]));

        FreeSpace(&(midhidden[i][j]));

        FreeSpace(&(poolmerge[i][j]));
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(pool[i][j][idm]));
          FreeSpace(&(poolIndex[i][j][idm]));
        }
      }
      FreeSpace(&(nbestmerge[i]));
    }
    FreeSpace(&sentmerge);
    FreeSpace(&project);
    FreeSpace(&output);

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    for (int i = 0; i < _segStylelabelAlphabet.size(); i++) {
      _seg_words[i].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _rnn_left[i].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _rnn_right[i].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _cnn_project[i].updateAdaGrad(nnRegular, adaAlpha, adaEps);
    }

  }

  void writeModel();

  void loadModel();

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(this, examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(this, examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(this, examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(this, examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    for (int i = 0; i < _segStylelabelAlphabet.size(); i++) {

      checkgrad(this, examples, _cnn_project[i]._W, _cnn_project[i]._gradW, _segStylelabelAlphabet.from_id(i) + "_cnn_project._W", iter);
      checkgrad(this, examples, _cnn_project[i]._b, _cnn_project[i]._gradb, _segStylelabelAlphabet.from_id(i) + "_cnn_project._b", iter);

      checkgrad(this, examples, _rnn_left[i]._lstm_output._W1, _rnn_left[i]._lstm_output._gradW1,
          _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_output._W1", iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_output._W2, _rnn_left[i]._lstm_output._gradW2,
          _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_output._W2", iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_output._W3, _rnn_left[i]._lstm_output._gradW3,
          _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_output._W3", iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_output._b, _rnn_left[i]._lstm_output._gradb, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_output._b",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_input._W1, _rnn_left[i]._lstm_input._gradW1, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_input._W1",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_input._W2, _rnn_left[i]._lstm_input._gradW2, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_input._W2",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_input._W3, _rnn_left[i]._lstm_input._gradW3, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_input._W3",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_input._b, _rnn_left[i]._lstm_input._gradb, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_input._b",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_forget._W1, _rnn_left[i]._lstm_forget._gradW1,
          _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_forget._W1", iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_forget._W2, _rnn_left[i]._lstm_forget._gradW2,
          _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_forget._W2", iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_forget._W3, _rnn_left[i]._lstm_forget._gradW3,
          _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_forget._W3", iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_forget._b, _rnn_left[i]._lstm_forget._gradb, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_forget._b",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_cell._WL, _rnn_left[i]._lstm_cell._gradWL, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_cell._WL",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_cell._WR, _rnn_left[i]._lstm_cell._gradWR, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_cell._WR",
          iter);
      checkgrad(this, examples, _rnn_left[i]._lstm_cell._b, _rnn_left[i]._lstm_cell._gradb, _segStylelabelAlphabet.from_id(i) + "_rnn_left._lstm_cell._b",
          iter);

      checkgrad(this, examples, _rnn_right[i]._lstm_output._W1, _rnn_right[i]._lstm_output._gradW1,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_output._W1", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_output._W2, _rnn_right[i]._lstm_output._gradW2,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_output._W2", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_output._W3, _rnn_right[i]._lstm_output._gradW3,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_output._W3", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_output._b, _rnn_right[i]._lstm_output._gradb,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_output._b", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_input._W1, _rnn_right[i]._lstm_input._gradW1,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_input._W1", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_input._W2, _rnn_right[i]._lstm_input._gradW2,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_input._W2", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_input._W3, _rnn_right[i]._lstm_input._gradW3,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_input._W3", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_input._b, _rnn_right[i]._lstm_input._gradb, _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_input._b",
          iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_forget._W1, _rnn_right[i]._lstm_forget._gradW1,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_forget._W1", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_forget._W2, _rnn_right[i]._lstm_forget._gradW2,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_forget._W2", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_forget._W3, _rnn_right[i]._lstm_forget._gradW3,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_forget._W3", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_forget._b, _rnn_right[i]._lstm_forget._gradb,
          _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_forget._b", iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_cell._WL, _rnn_right[i]._lstm_cell._gradWL, _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_cell._WL",
          iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_cell._WR, _rnn_right[i]._lstm_cell._gradWR, _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_cell._WR",
          iter);
      checkgrad(this, examples, _rnn_right[i]._lstm_cell._b, _rnn_right[i]._lstm_cell._gradb, _segStylelabelAlphabet.from_id(i) + "_rnn_right._lstm_cell._b",
          iter);

      checkgrad(this, examples, _seg_words[i]._E, _seg_words[i]._gradE, _segStylelabelAlphabet.from_id(i) + "_words._E", iter, _seg_words[i]._indexers);
    }

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {

    for (int i = 0; i < _segStylelabelAlphabet.size(); i++) {
      _seg_words[i].setEmbFineTune(b_wordEmb_finetune);
    }
  }

  inline void resetRemove(int remove) {
    _remove = remove;
  }

};

#endif /* SRC_GRCNNWordClassifier_H_ */
