#ifndef __ARCHIVE_IO_H
#define __ARCHIVE_IO_H

#include <pbar.h>
#include <utility.h>
#include <array.h>

#include <vulcan-hmm.h>
#include <vulcan-archive.h>

#include <libutility/include/utility.h>
#include <libfeature/include/feature.h>

using namespace vulcan;

typedef vector<DoubleVector> FeatureSeq;
typedef std::pair<size_t, size_t> Phone; 

#define check_equal(a, b) { \
  cout << "Checking equivalence... " \
  << #a << "(" << a << ")" \
  << ((a == b) ? GREEN " == " COLOREND : ORANGE " != " COLOREND) \
  << #b << "(" << b << ")" << endl; };

int load(string alignmentFile, string modelFile, map<string, vector<Phone> >& phoneLabels, bool dump = false);
size_t loadFeatureArchive(const string& featArk, const map<string, vector<Phone> >& phoneLabels, map<size_t, vector<FeatureSeq> >& phoneInstances);
void loadFeatureArchive(string filename, float* &data, unsigned int* &offset, int& N, int& dim);

void save(const FeatureSeq& featureSeq, const string& filename);
size_t saveFeatureAsMFCC(const map<size_t, vector<FeatureSeq> >& phoneInstances, const vector<string>& phones, string dir);

vector<string> getPhoneMapping(string filename);
void print(FILE* p, const FeatureSeq& fs);

#endif // __ARCHIVE_IO_H
