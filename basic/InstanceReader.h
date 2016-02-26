#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "MyLib.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader: public Reader {
public:
  InstanceReader() {
  }
  ~InstanceReader() {
  }

  Instance *getNext() {
    m_instance.clear();

    vector<string> vecLine;
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
    int labelsize = m_segStylelabelAlphabet.size();

    if (length == 1) {
      m_instance.allocate(1);
      vector<string> vecInfo;
      split_bychar(vecLine[0], vecInfo, ' ');
      int veclength = vecInfo.size();
      m_instance.label = vecInfo[0];
      vector<string> sent;
      for (int j = 1; j < veclength; j++) {
        sent.push_back(vecInfo[j]);
        //vector<string> curChars;
        //getCharactersFromUTF8String(vecInfo[j], curChars);
        //m_instance.chars[0].push_back(curChars);
      }
      m_instance.words[0].push_back(sent);
    } else {
      m_instance.allocate(labelsize);
      for (int i = 0; i < length; ++i) {
        vector<string> vecInfo;
        split_bychar(vecLine[i], vecInfo, ' ');
        int veclength = vecInfo.size();
        if (i == length - 1) {
          m_instance.label = vecInfo[0];
        } else {
          int labelId = m_segStylelabelAlphabet.from_string(vecInfo[0]);
          vector<string> nbestsent;
          for (int j = 1; j < veclength; j++) {
            nbestsent.push_back(vecInfo[j]);
            //vector<string> curChars;
            //getCharactersFromUTF8String(vecInfo[j], curChars);
            //m_instance.chars[labelId].push_back(curChars);

          }
          m_instance.words[labelId].push_back(nbestsent);

        }

      }
    }

    return &m_instance;
  }
public:
};

#endif

