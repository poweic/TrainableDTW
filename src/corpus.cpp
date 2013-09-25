#include <corpus.h>
#include <iostream>
#include <array.h>
using namespace std;

const string SubCorpus::MFCC_DIRECTORY = "data/mfcc/";
const string SubCorpus::LIST_DIRECTORY = "data/train/list/";

SubCorpus::SubCorpus() {
}

SubCorpus::SubCorpus(size_t p1, size_t p2, string phone1, string phone2): _p1(p1), _p2(p2), _phone1(phone1), _phone2(phone2), _counter(0) {
  _init();
}

vector<ppair> SubCorpus::getSamples(size_t n) {
  if (n <= 0) return vector<ppair>();

  vector<ppair> samples;
  samples.reserve(n);

  size_t c = 0;
  foreach (i, _list1) {
    foreach (j, _list2) {
      if (c++ < _counter)
	continue;

      if (c >= _counter + n)
	break;

      string f1 = MFCC_DIRECTORY + _phone1 + "/" + _list1[i];
      string f2 = MFCC_DIRECTORY + _phone2 + "/" + _list2[j];
      samples.push_back(ppair(f1, f2));
    }
  }
  _counter += samples.size();
  if (_counter >= _size)
    _counter -= _size;

  return samples;
}

bool SubCorpus::isIntraPhone() const { return (_p1 == _p2); }

void SubCorpus::_init() {
  _list1 = Array<string>(SubCorpus::LIST_DIRECTORY + int2str(_p1) + ".list");
  _list2 = Array<string>(SubCorpus::LIST_DIRECTORY + int2str(_p2) + ".list");

  if (_p1 == _p2)
    _size = _list1.size() * (_list1.size() - 1) / 2;
  else
    _size = _list1.size() * _list2.size();
}


// ==============================================
Corpus::Corpus(string filename) {
    _phones = this->getPhoneList(filename);

    size_t size = 0;
    // Load all lists for 74 phones
    foreach (i, _phones) {
      if (i <= 1) continue;
      foreach (j, _phones) {
	if (j <= 1) continue;
	if (j > i) break;
	
	//cout << "i = " << i << ", j = " << j << endl;
	_sub_corpus.push_back(SubCorpus(i, j, _phones[i], _phones[j]));
	size += _sub_corpus.back().size();
      }
    }
    _size = size;
  }

  Array<string> Corpus::getPhoneList(string filename) {

    Array<string> list;

    ifstream file(filename);
    string line;
    while( std::getline(file, line) ) {
      vector<string> sub = split(line, ' ');
      string phone = sub[0];
      list.push_back(phone);
    }
    file.close();

    return list;
  }

  vector<tsample> Corpus::getSamples(size_t n) {
    if (n <= 0) return vector<tsample>();

    vector<tsample> samples;
    samples.reserve(n);

    foreach (i, _sub_corpus) {
      int nSubSamples = (double) _sub_corpus[i].size() / (double) this->size() * (double) n;
      vector<ppair> subSamples = _sub_corpus[i].getSamples(nSubSamples);
      bool positive = _sub_corpus[i].isIntraPhone();
      foreach (j, subSamples)
	samples.push_back(tsample(subSamples[j], positive));
      //samples.insert(samples.end(), subSamples.begin(), subSamples.end());
    }

    return samples;
  }

  bool Corpus::isBatchSizeApprop(size_t batchSize) {
    vector<tsample> samples = this->getSamples(batchSize);

    double RATE = 0.3;
    if ( (double) samples.size() / (double) batchSize < RATE ) {
      cerr << "Batch size ( = " << batchSize << " ) too small" << endl
	<< "Only " << samples.size() << " data are sampled from the corpus"
	<< ", leading to extremely bad estimation of distribution." << endl;
      return false;
    }

    return true;
  }
