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

  LookupTable<xpu> _words;

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

  LSTM_STD<xpu> _rnn_left;
  LSTM_STD<xpu> _rnn_right;
  UniLayer<xpu> _cnn_project;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3 min

  int _poolmanners;

  Alphabet _segStylelabelAlphabet;

public:

  inline void init(const NRMat<dtype>& wordEmb, int wordcontext, int labelSize, int wordHiddenSize, int hiddenSize) {
    _wordcontext = wordcontext;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();
    _poolmanners = 3;

    _labelSize = labelSize;
    _hiddenSize = hiddenSize;
    _wordHiddenSize = wordHiddenSize;
    _token_representation_size = _wordDim;

    _word_cnn_iSize = _token_representation_size * (2 * _wordcontext + 1);

    _rnn_left.initial(_wordHiddenSize, _word_cnn_iSize, true, 40);
    _rnn_right.initial(_wordHiddenSize, _word_cnn_iSize, false, 70);
    _cnn_project.initial(_wordHiddenSize, 2 * _wordHiddenSize, true, 90, 0);

    _words.initial(wordEmb);

    _tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize, true, 50, 0);
    _olayer_linear.initial(_labelSize, hiddenSize, false, 60, 2);

    _eval.reset();

    _remove = 0;

  }

  inline void release() {

    _rnn_left.release();
    _rnn_right.release();
    _cnn_project.release();

    _words.release();
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

      vector<vector<Tensor<xpu, 2, dtype> > > poolnBest, poolnBestLoss;
      vector<vector<vector<Tensor<xpu, 3, dtype> > > > poolIndex;
      vector<Tensor<xpu, 2, dtype> > poolnBestmerge, poolnBestmergeLoss;
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
      poolnBest.resize(seg_style);
      poolnBestLoss.resize(seg_style);
      poolIndex.resize(seg_style);

      poolnBestmerge.resize(seg_style);
      poolnBestmergeLoss.resize(seg_style);

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
        poolnBest[i].resize(_poolmanners);
        poolnBestLoss[i].resize(_poolmanners);

        poolIndex[i].resize(_poolmanners);

        for (int j = 0; j < _poolmanners; j++) {
          poolIndex[i][j].resize(nbest_num);
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

          hidden[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          hiddenLoss[i][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

          for (int idm = 0; idm < _poolmanners; idm++) {
            poolIndex[i][idm][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          }

        }

        for (int idm = 0; idm < _poolmanners; idm++) {
          poolnBest[i][idm] = NewTensor<xpu>(Shape2(1, _wordHiddenSize), d_zero);
          poolnBestLoss[i][idm] = NewTensor<xpu>(Shape2(1, _wordHiddenSize), d_zero);
        }

        poolnBestmerge[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
        poolnBestmergeLoss[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);

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
            _words.GetEmb(words[idy], wordprime[i][j][idy]);
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

          _rnn_left.ComputeForwardScore(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
              rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j]);
          _rnn_right.ComputeForwardScore(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j], rnn_hidden_right_mc[i][j],
              rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j]);

          for (int idy = 0; idy < word_num; idy++) {
            concat(rnn_hidden_left[i][j][idy], rnn_hidden_right[i][j][idy], midhidden[i][j][idy]);
          }

          _cnn_project.ComputeForwardScore(midhidden[i][j], hidden[i][j]);

        }
      }

      //min pool
      for (int i = 0; i < seg_style; i++) {
        int minpoolId = 0;
        for (int d = 0; d < _wordHiddenSize; d++) {
          dtype min = 0;
          int minj = -1, mink = -1;
          int nbest_num = hidden[i].size();
          for (int j = 0; j < nbest_num; j++) {
            int word_num = hidden[i][j].size(0);
            for (int k = 0; k < word_num; k++) {
              if (minj < 0 || hidden[i][j][k][0][d] < min) {
                min = hidden[i][j][k][0][d];
                mink = k;
                minj = j;
              }
            }
          }
          poolnBest[i][minpoolId][0][d] = min;
          poolIndex[i][minpoolId][minj][mink][0][d] = 1;
        }
      }
      //maxpool
      for (int i = 0; i < seg_style; i++) {
        int maxpoolId = 1;
        for (int d = 0; d < _wordHiddenSize; d++) {
          dtype max = 0;
          int maxj = -1, maxk = -1;
          int nbest_num = hidden[i].size();
          for (int j = 0; j < nbest_num; j++) {
            int word_num = hidden[i][j].size(0);
            for (int k = 0; k < word_num; k++) {
              if (maxj < 0 || hidden[i][j][k][0][d] > max) {
                max = hidden[i][j][k][0][d];
                maxk = k;
                maxj = j;
              }
            }
          }
          poolnBest[i][maxpoolId][0][d] = max;
          poolIndex[i][maxpoolId][maxj][maxk][0][d] = 1;
        }
      }
      //avgpool
      for (int i = 0; i < seg_style; i++) {

        int total_num = 0;
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          int word_num = hidden[i][j].size(0);
          total_num += word_num;
        }

        int avgpoolID = 2;
        for (int d = 0; d < _wordHiddenSize; d++) {
          dtype sum = 0.0;
          int nbest_num = hidden[i].size();
          for (int j = 0; j < nbest_num; j++) {
            int word_num = hidden[i][j].size(0);
            for (int k = 0; k < word_num; k++) {
              sum += hidden[i][j][k][0][d];
              poolIndex[i][avgpoolID][j][k][0][d] = 1.0 / total_num;

            }
          }
          poolnBest[i][avgpoolID][0][d] = sum / total_num;
        }
      }

      for (int i = 0; i < seg_style; i++) {
        concat(poolnBest[i], poolnBestmerge[i]);
      }
      concat(poolnBestmerge, sentmerge);
      _tanh_project.ComputeForwardScore(sentmerge, project);
      _olayer_linear.ComputeForwardScore(project, output);

      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      //sentence
      _olayer_linear.ComputeBackwardLoss(project, output, outputLoss, projectLoss);
      _tanh_project.ComputeBackwardLoss(sentmerge, project, projectLoss, sentmergeLoss);

      unconcat(poolnBestmergeLoss, sentmergeLoss);
      for (int i = 0; i < seg_style; i++) {
        unconcat(poolnBestLoss[i], poolnBestmergeLoss[i]);
      }

      //word pooling
      for (int i = 0; i < seg_style; i++) {
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            pool_backward(poolnBestLoss[i][0], poolIndex[i][0][j], hiddenLoss[i][j]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            pool_backward(poolnBestLoss[i][1], poolIndex[i][1][j], hiddenLoss[i][j]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            pool_backward(poolnBestLoss[i][2], poolIndex[i][2][j], hiddenLoss[i][j]);
          }
        }
      }

      for (int i = 0; i < seg_style; i++) {
        int nbest_num = example.m_features[i].size();
        for (int j = 0; j < nbest_num; j++) {

          const Feature& feature = example.m_features[i][j];
          int curcontext = _wordcontext;
          const vector<int>& words = feature.words;
          int word_num = words.size();

          _cnn_project.ComputeBackwardLoss(midhidden[i][j], hidden[i][j], hiddenLoss[i][j], midhiddenLoss[i][j]);

          for (int idy = 0; idy < word_num; idy++) {
            unconcat(rnn_hidden_leftLoss[i][j][idy], rnn_hidden_rightLoss[i][j][idy], midhiddenLoss[i][j][idy]);
          }

          _rnn_left.ComputeBackwardLoss(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
              rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j], rnn_hidden_leftLoss[i][j], inputLoss[i][j]);
          _rnn_right.ComputeBackwardLoss(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j], rnn_hidden_right_mc[i][j],
              rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j], rnn_hidden_rightLoss[i][j], inputLoss[i][j]);

          windowlized_backward(wordrepresentLoss[i][j], inputLoss[i][j], curcontext);

          //word representation
          for (int idy = 0; idy < word_num; idy++) {
            wordprimeLoss[i][j][idy] += wordrepresentLoss[i][j][idy];
          }

          if (_words.bEmbFineTune()) {
            for (int idy = 0; idy < word_num; idy++) {
              wordprimeLoss[i][j][idy] = wordprimeLoss[i][j][idy] * wordprimeMask[i][j][idy];
              _words.EmbLoss(words[idy], wordprimeLoss[i][j][idy]);
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
          FreeSpace(&(hidden[i][j]));
          FreeSpace(&(hiddenLoss[i][j]));

          for (int idm = 0; idm < _poolmanners; idm++) {
            FreeSpace(&(poolIndex[i][idm][j]));
          }
        }
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(poolnBest[i][idm]));
          FreeSpace(&(poolnBestLoss[i][idm]));
        }
        FreeSpace(&(poolnBestmerge[i]));
        FreeSpace(&(poolnBestmergeLoss[i]));
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

    vector<vector<Tensor<xpu, 2, dtype> > > poolnBest;
    vector<vector<vector<Tensor<xpu, 3, dtype> > > > poolIndex;
    vector<Tensor<xpu, 2, dtype> > poolnBestmerge;

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
    poolnBest.resize(seg_style);
    poolIndex.resize(seg_style);

    poolnBestmerge.resize(seg_style);

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

      poolnBest[i].resize(_poolmanners);

      poolIndex[i].resize(_poolmanners);

      for (int j = 0; j < _poolmanners; j++) {
        poolIndex[i][j].resize(nbest_num);
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
          poolIndex[i][idm][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        }

      }

      for (int idm = 0; idm < _poolmanners; idm++) {
        poolnBest[i][idm] = NewTensor<xpu>(Shape2(1, _wordHiddenSize), d_zero);
      }

      poolnBestmerge[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
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
          _words.GetEmb(words[idy], wordprime[i][j][idy]);
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          wordrepresent[i][j][idy] += wordprime[i][j][idy];
        }

        windowlized(wordrepresent[i][j], input[i][j], curcontext);

        _rnn_left.ComputeForwardScore(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
            rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j]);
        _rnn_right.ComputeForwardScore(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j], rnn_hidden_right_mc[i][j],
            rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j]);

        for (int idy = 0; idy < word_num; idy++) {
          concat(rnn_hidden_left[i][j][idy], rnn_hidden_right[i][j][idy], midhidden[i][j][idy]);
        }

        _cnn_project.ComputeForwardScore(midhidden[i][j], hidden[i][j]);

      }
    }

    //min pool
    for (int i = 0; i < seg_style; i++) {
      int minpoolId = 0;
      for (int d = 0; d < _wordHiddenSize; d++) {
        dtype min = 0;
        int minj = -1, mink = -1;
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          int word_num = hidden[i][j].size(0);
          for (int k = 0; k < word_num; k++) {
            if (minj < 0 || hidden[i][j][k][0][d] < min) {
              min = hidden[i][j][k][0][d];
              mink = k;
              minj = j;
            }
          }
        }
        poolnBest[i][minpoolId][0][d] = min;
        poolIndex[i][minpoolId][minj][mink][0][d] = 1;
      }
    }
    //maxpool
    for (int i = 0; i < seg_style; i++) {
      int maxpoolId = 1;
      for (int d = 0; d < _wordHiddenSize; d++) {
        dtype max = 0;
        int maxj = -1, maxk = -1;
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          int word_num = hidden[i][j].size(0);
          for (int k = 0; k < word_num; k++) {
            if (maxj < 0 || hidden[i][j][k][0][d] > max) {
              max = hidden[i][j][k][0][d];
              maxk = k;
              maxj = j;
            }
          }
        }
        poolnBest[i][maxpoolId][0][d] = max;
        poolIndex[i][maxpoolId][maxj][maxk][0][d] = 1;
      }
    }
    //avgpool
    for (int i = 0; i < seg_style; i++) {

      int total_num = 0;
      int nbest_num = hidden[i].size();
      for (int j = 0; j < nbest_num; j++) {
        int word_num = hidden[i][j].size(0);
        total_num += word_num;
      }

      int avgpoolID = 2;
      for (int d = 0; d < _wordHiddenSize; d++) {
        dtype sum = 0.0;
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          int word_num = hidden[i][j].size(0);
          for (int k = 0; k < word_num; k++) {
            sum += hidden[i][j][k][0][d];
            poolIndex[i][avgpoolID][j][k][0][d] = 1.0 / total_num;

          }
        }
        poolnBest[i][avgpoolID][0][d] = sum / total_num;
      }
    }

    for (int i = 0; i < seg_style; i++) {
      concat(poolnBest[i], poolnBestmerge[i]);
    }
    concat(poolnBestmerge, sentmerge);
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
        FreeSpace(&(hidden[i][j]));

        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(poolIndex[i][idm][j]));
        }
      }
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(poolnBest[i][idm]));
      }
      FreeSpace(&(poolnBestmerge[i]));
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

    vector<vector<Tensor<xpu, 2, dtype> > > poolnBest;
    vector<vector<vector<Tensor<xpu, 3, dtype> > > > poolIndex;
    vector<Tensor<xpu, 2, dtype> > poolnBestmerge;

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
    poolnBest.resize(seg_style);
    poolIndex.resize(seg_style);

    poolnBestmerge.resize(seg_style);

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

      poolnBest[i].resize(_poolmanners);

      poolIndex[i].resize(_poolmanners);

      for (int j = 0; j < _poolmanners; j++) {
        poolIndex[i][j].resize(nbest_num);
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
          poolIndex[i][idm][j] = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        }
      }

      for (int idm = 0; idm < _poolmanners; idm++) {
        poolnBest[i][idm] = NewTensor<xpu>(Shape2(1, _wordHiddenSize), d_zero);
      }

      poolnBestmerge[i] = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
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
          _words.GetEmb(words[idy], wordprime[i][j][idy]);
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          wordrepresent[i][j][idy] += wordprime[i][j][idy];
        }

        windowlized(wordrepresent[i][j], input[i][j], curcontext);

        _rnn_left.ComputeForwardScore(input[i][j], rnn_hidden_left_i[i][j], rnn_hidden_left_o[i][j], rnn_hidden_left_f[i][j], rnn_hidden_left_mc[i][j],
            rnn_hidden_left_c[i][j], rnn_hidden_left_my[i][j], rnn_hidden_left[i][j]);
        _rnn_right.ComputeForwardScore(input[i][j], rnn_hidden_right_i[i][j], rnn_hidden_right_o[i][j], rnn_hidden_right_f[i][j], rnn_hidden_right_mc[i][j],
            rnn_hidden_right_c[i][j], rnn_hidden_right_my[i][j], rnn_hidden_right[i][j]);

        for (int idy = 0; idy < word_num; idy++) {
          concat(rnn_hidden_left[i][j][idy], rnn_hidden_right[i][j][idy], midhidden[i][j][idy]);
        }

        _cnn_project.ComputeForwardScore(midhidden[i][j], hidden[i][j]);

      }
    }

    //min pool
    for (int i = 0; i < seg_style; i++) {
      int minpoolId = 0;
      for (int d = 0; d < _wordHiddenSize; d++) {
        dtype min = 0;
        int minj = -1, mink = -1;
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          int word_num = hidden[i][j].size(0);
          for (int k = 0; k < word_num; k++) {
            if (minj < 0 || hidden[i][j][k][0][d] < min) {
              min = hidden[i][j][k][0][d];
              mink = k;
              minj = j;
            }
          }
        }
        poolnBest[i][minpoolId][0][d] = min;
        poolIndex[i][minpoolId][minj][mink][0][d] = 1;
      }
    }
    //maxpool
    for (int i = 0; i < seg_style; i++) {
      int maxpoolId = 1;
      for (int d = 0; d < _wordHiddenSize; d++) {
        dtype max = 0;
        int maxj = -1, maxk = -1;
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          int word_num = hidden[i][j].size(0);
          for (int k = 0; k < word_num; k++) {
            if (maxj < 0 || hidden[i][j][k][0][d] > max) {
              max = hidden[i][j][k][0][d];
              maxk = k;
              maxj = j;
            }
          }
        }
        poolnBest[i][maxpoolId][0][d] = max;
        poolIndex[i][maxpoolId][maxj][maxk][0][d] = 1;
      }
    }
    //avgpool
    for (int i = 0; i < seg_style; i++) {

      int total_num = 0;
      int nbest_num = hidden[i].size();
      for (int j = 0; j < nbest_num; j++) {
        int word_num = hidden[i][j].size(0);
        total_num += word_num;
      }

      int avgpoolID = 2;
      for (int d = 0; d < _wordHiddenSize; d++) {
        dtype sum = 0.0;
        int nbest_num = hidden[i].size();
        for (int j = 0; j < nbest_num; j++) {
          int word_num = hidden[i][j].size(0);
          for (int k = 0; k < word_num; k++) {
            sum += hidden[i][j][k][0][d];
            poolIndex[i][avgpoolID][j][k][0][d] = 1.0 / total_num;

          }
        }
        poolnBest[i][avgpoolID][0][d] = sum / total_num;
      }
    }

    for (int i = 0; i < seg_style; i++) {
      concat(poolnBest[i], poolnBestmerge[i]);
    }
    concat(poolnBestmerge, sentmerge);
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
        FreeSpace(&(hidden[i][j]));

        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(poolIndex[i][idm][j]));
        }
      }
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(poolnBest[i][idm]));
      }
    }
    FreeSpace(&sentmerge);
    FreeSpace(&project);
    FreeSpace(&output);

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);

  }

  void writeModel();

  void loadModel();

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(this, examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(this, examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(this, examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(this, examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(this, examples, _cnn_project._W, _cnn_project._gradW, "_cnn_project._W", iter);
    checkgrad(this, examples, _cnn_project._b, _cnn_project._gradb, "_cnn_project._b", iter);

    checkgrad(this, examples, _rnn_left._lstm_output._W1, _rnn_left._lstm_output._gradW1, "_rnn_left._lstm_output._W1", iter);
    checkgrad(this, examples, _rnn_left._lstm_output._W2, _rnn_left._lstm_output._gradW2, "_rnn_left._lstm_output._W2", iter);
    checkgrad(this, examples, _rnn_left._lstm_output._W3, _rnn_left._lstm_output._gradW3, "_rnn_left._lstm_output._W3", iter);
    checkgrad(this, examples, _rnn_left._lstm_output._b, _rnn_left._lstm_output._gradb, "_rnn_left._lstm_output._b", iter);
    checkgrad(this, examples, _rnn_left._lstm_input._W1, _rnn_left._lstm_input._gradW1, "_rnn_left._lstm_input._W1", iter);
    checkgrad(this, examples, _rnn_left._lstm_input._W2, _rnn_left._lstm_input._gradW2, "_rnn_left._lstm_input._W2", iter);
    checkgrad(this, examples, _rnn_left._lstm_input._W3, _rnn_left._lstm_input._gradW3, "_rnn_left._lstm_input._W3", iter);
    checkgrad(this, examples, _rnn_left._lstm_input._b, _rnn_left._lstm_input._gradb, "_rnn_left._lstm_input._b", iter);
    checkgrad(this, examples, _rnn_left._lstm_forget._W1, _rnn_left._lstm_forget._gradW1, "_rnn_left._lstm_forget._W1", iter);
    checkgrad(this, examples, _rnn_left._lstm_forget._W2, _rnn_left._lstm_forget._gradW2, "_rnn_left._lstm_forget._W2", iter);
    checkgrad(this, examples, _rnn_left._lstm_forget._W3, _rnn_left._lstm_forget._gradW3, "_rnn_left._lstm_forget._W3", iter);
    checkgrad(this, examples, _rnn_left._lstm_forget._b, _rnn_left._lstm_forget._gradb, "_rnn_left._lstm_forget._b", iter);
    checkgrad(this, examples, _rnn_left._lstm_cell._WL, _rnn_left._lstm_cell._gradWL, "_rnn_left._lstm_cell._WL", iter);
    checkgrad(this, examples, _rnn_left._lstm_cell._WR, _rnn_left._lstm_cell._gradWR, "_rnn_left._lstm_cell._WR", iter);
    checkgrad(this, examples, _rnn_left._lstm_cell._b, _rnn_left._lstm_cell._gradb, "_rnn_left._lstm_cell._b", iter);

    checkgrad(this, examples, _rnn_right._lstm_output._W1, _rnn_right._lstm_output._gradW1, "_rnn_right._lstm_output._W1", iter);
    checkgrad(this, examples, _rnn_right._lstm_output._W2, _rnn_right._lstm_output._gradW2, "_rnn_right._lstm_output._W2", iter);
    checkgrad(this, examples, _rnn_right._lstm_output._W3, _rnn_right._lstm_output._gradW3, "_rnn_right._lstm_output._W3", iter);
    checkgrad(this, examples, _rnn_right._lstm_output._b, _rnn_right._lstm_output._gradb, "_rnn_right._lstm_output._b", iter);
    checkgrad(this, examples, _rnn_right._lstm_input._W1, _rnn_right._lstm_input._gradW1, "_rnn_right._lstm_input._W1", iter);
    checkgrad(this, examples, _rnn_right._lstm_input._W2, _rnn_right._lstm_input._gradW2, "_rnn_right._lstm_input._W2", iter);
    checkgrad(this, examples, _rnn_right._lstm_input._W3, _rnn_right._lstm_input._gradW3, "_rnn_right._lstm_input._W3", iter);
    checkgrad(this, examples, _rnn_right._lstm_input._b, _rnn_right._lstm_input._gradb, "_rnn_right._lstm_input._b", iter);
    checkgrad(this, examples, _rnn_right._lstm_forget._W1, _rnn_right._lstm_forget._gradW1, "_rnn_right._lstm_forget._W1", iter);
    checkgrad(this, examples, _rnn_right._lstm_forget._W2, _rnn_right._lstm_forget._gradW2, "_rnn_right._lstm_forget._W2", iter);
    checkgrad(this, examples, _rnn_right._lstm_forget._W3, _rnn_right._lstm_forget._gradW3, "_rnn_right._lstm_forget._W3", iter);
    checkgrad(this, examples, _rnn_right._lstm_forget._b, _rnn_right._lstm_forget._gradb, "_rnn_right._lstm_forget._b", iter);
    checkgrad(this, examples, _rnn_right._lstm_cell._WL, _rnn_right._lstm_cell._gradWL, "_rnn_right._lstm_cell._WL", iter);
    checkgrad(this, examples, _rnn_right._lstm_cell._WR, _rnn_right._lstm_cell._gradWR, "_rnn_right._lstm_cell._WR", iter);
    checkgrad(this, examples, _rnn_right._lstm_cell._b, _rnn_right._lstm_cell._gradb, "_rnn_right._lstm_cell._b", iter);

    checkgrad(this, examples, _words._E, _words._gradE, +"_words._E", iter, _words._indexers);

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {

    _words.setEmbFineTune(b_wordEmb_finetune);
  }

  inline void resetRemove(int remove) {
    _remove = remove;
  }

};

#endif /* SRC_GRCNNWordClassifier_H_ */
