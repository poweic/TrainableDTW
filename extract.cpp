#include <string>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <map>

#include <archive_io.h>
#include <cmdparser.h>

using namespace std;

int main(int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .addGroup("Generic options:")
    .add("-a", "filename of alignments")
    .add("-p", "phone table")
    .add("-m", "model");

  cmdParser
    .addGroup("Other options:")
    .add("--feat-ark", "feature archive where mfcc extracted from", false);

  cmdParser
    .addGroup("Examples: ./extract -a data/train.ali.txt -p data/phones.txt -m data/final.mdl --feat-ark=/media/Data1/LectureDSP_script/feat/test.39.ark");

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  string alignmentFile = cmdParser.find("-a");
  string phoneTableFile = cmdParser.find("-p");
  string modelFile = cmdParser.find("-m");
  string featArk = cmdParser.find("--feat-akr");

  debug(alignmentFile);
  debug(phoneTableFile);
  debug(modelFile);

  vector<string> phones = getPhoneMapping(phoneTableFile);

  map<string, vector<Phone> > phoneLabels;
  int nInstance = load(alignmentFile, modelFile, phoneLabels);

  map<size_t, vector<FeatureSeq> > phoneInstances;

  if (featArk.empty())
    return 0;

  int n = loadFeatureArchive(featArk, phoneLabels, phoneInstances);
  check_equal(n, nInstance);

  size_t nMfccFiles = saveFeatureAsMFCC(phoneInstances, phones);
  check_equal(nMfccFiles, nInstance);

  cout << "[Done]" << endl;

  return 0;
}


