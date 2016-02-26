#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "Instance.h"
#include "N3L.h"

class Reader {
public:
  Reader() {
  }

  virtual ~Reader() {
    if (m_inf.is_open())
      m_inf.close();
  }
  int startReading(const char *filename) {
    if (m_inf.is_open()) {
      m_inf.close();
      m_inf.clear();
    }
    m_inf.open(filename);
    /********************************/
    m_segStylelabelAlphabet.clear();
    vector<string> vecLine;
    int labelId;
    while (1) {
      string strLine;
      if (!my_getline(m_inf, strLine)) {
        break;
      }
      if (strLine.empty())
        break;
      vecLine.push_back(strLine);
    }

    int length = vecLine.size();

    if (length > 1) {
      for (int i = 0; i < length; ++i) {
        vector<string> vecInfo;
        split_bychar(vecLine[i], vecInfo, ' ');
        int veclength = vecInfo.size();
        if (i == length - 1) {
          if (vecInfo.size() != 1) {
            std::cerr << "input format error: the last line is not label" << std::endl;
          }
        } else {

          labelId = m_segStylelabelAlphabet.from_string(vecInfo[0]);
        }
      }
    }
    /********************************/

    m_inf.close();
    m_inf.clear();
    m_inf.open(filename);

    if (!m_inf.is_open()) {
      cout << "Reader::startReading() open file err: " << filename << endl;
      return -1;
    }

    return 0;
  }

  void finishReading() {
    if (m_inf.is_open()) {
      m_inf.close();
      m_inf.clear();
    }
  }
  virtual Instance *getNext() = 0;
protected:
  ifstream m_inf;

  int m_numInstance;

  Instance m_instance;
public:
  Alphabet m_segStylelabelAlphabet;
};

#endif

