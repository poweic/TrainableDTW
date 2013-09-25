#ifndef __CORPUS_H_
#define __CORPUS_H_

#include <string>
#include <vector>
#include <array.h>
#include <utility.h>
using namespace std;

template <typename T>
class ICorpus {
public:
  virtual vector<T> getSamples(size_t n) = 0;
  virtual size_t size() const { return _size; }
protected:
  size_t _size;
};

typedef pair<string, string> ppair;

class SubCorpus : public ICorpus<ppair> {
public:
  SubCorpus(size_t p1, size_t p2, string phone1, string phone2);

  vector<ppair> getSamples(size_t n);

  bool isIntraPhone() const;

  static const string LIST_DIRECTORY;
  static const string MFCC_DIRECTORY;

private:
  SubCorpus();

  void _init();

  size_t _counter;

  size_t _p1;
  size_t _p2;
  string _phone1;
  string _phone2;
  Array<string> _list1;
  Array<string> _list2;
};

typedef std::pair<ppair, bool> tsample;

class Corpus: public ICorpus<tsample> {
public:
  Corpus(string filename);

  Array<string> getPhoneList(string filename);

  vector<tsample> getSamples(size_t n);

  bool isBatchSizeApprop(size_t batchSize);

private:
  Array<string> _phones;
  vector<SubCorpus> _sub_corpus;
};

#endif // __CORPUS_H_
