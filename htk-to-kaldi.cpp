#include <iostream>

#include <cmdparser.h>
#include <array.h>
#include <matrix.h>
#include <util.h>
#include <utility.h>
#include <profile.h>
#include <trainable_dtw.h>

using namespace DtwUtil;
using namespace std;

void dumpMfccAsKaldiArk(const Array<string>& lists);
void chooseLargestGranularity(const string& path, Array<string>& lists, string file_extension);

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .add("-d", "directory of HTK feature files")
    .add("--list", "list of feature filenames")
    .add("--extension", "choose either \".mfc\" or \".gp\"");

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  string path = cmdParser.find("-d") + "/";
  string list_filename = cmdParser.find("--list");
  string file_extension = cmdParser.find("--extension");

  Array<string> lists(list_filename);
  chooseLargestGranularity(path, lists, file_extension);

  dumpMfccAsKaldiArk(lists);
  return 0;
}

void dumpMfccAsKaldiArk(const Array<string>& lists) {

  foreach (i, lists) {
    cout << lists[i] << "  [" << endl;

    DtwParm p(lists[i]);
    size_t feat_dim = p.Feat().LF();
    size_t totalTime = p.Feat().LT();
    range (t, totalTime) {
      cout << "  ";

      range (d, feat_dim)
	cout << p.Feat()[t][d] << " ";

      if (t == totalTime - 1)
	cout << "]";
      cout << endl;
    }
  }
}

void chooseLargestGranularity(const string& path, Array<string>& lists, string file_extension) {

  if (file_extension[0] != '.')
    file_extension = "." + file_extension;
  
  foreach (i, lists) {
    string filename = path + lists[i] + file_extension;
    if (exists(filename)) {
      lists[i] = filename;
      continue;
    }

    // Choose Highest number. (i.e. largest granularity)
    // Granularity: word > character > syllable > phone
    for (int j=1; j<50; ++j) {
      string filename = path + lists[i] + "_" + int2str(j) + file_extension;
      if (exists(filename)) {
	lists[i] = filename;
	break;
      }
    }

  }
}
