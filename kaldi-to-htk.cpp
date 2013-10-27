#include <string>
#include <iostream>
#include <array.h>

#include <archive_io.h>
#include <cmdparser.h>

void saveFeatureArchiveAsHtk(const string& featArk);

using namespace std;

int main(int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .addGroup("Generic options:")
    .add("-q", "filename of the query list")
    .add("-f", "folder containing all the feature archive");

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  string ql_fn = cmdParser.find("-q");
  string archive = cmdParser.find("-f");

  Array<string> queries(ql_fn);
  
  for (size_t i=0; i<queries.size(); ++i) {
    string fn = archive + "/" + queries[i] + ".76.ark";
    cout << fn << endl;
    saveFeatureArchiveAsHtk(fn);
  }
  return 0;
}

void saveFeatureArchiveAsHtk(const string& featArk) {

  FILE* fptr = fopen(featArk.c_str(), "r");
  VulcanUtterance vUtterance;
  while (vUtterance.LoadKaldi(fptr)) {
    const FeatureSeq& fs = vUtterance._feature;
    string docId = vUtterance.fid();

    docId = replace_all(docId, "mfc", "gp");
    docId = replace_all(docId, "SI_word", "SI_word_gp");
    docId = replace_all(docId, "OOV_g2p", "OOV_g2p_gp");

    // cout << docId << endl;
    save(fs, docId);
  }
}

