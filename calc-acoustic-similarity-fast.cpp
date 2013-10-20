#include <iostream>
#include <matrix.h>
#include <perf.h>
#include <cmdparser.h>

#include <fast_dtw.h>

using namespace std;

void normalize(float** m, int N);

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .add("--ark", "input feature archive")
    .add("-o", "output filename for the acoustic similarity matrix");

  cmdParser
    .addGroup("Distance options")
    .add("--theta", "specify the file containing the diagnol term of Mahalanobis distance (dim=39)", false)
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-4");
  
  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  perf::Timer timer;
  timer.start();

  // string fn = "/media/Data1/hypothesis/SI_word.kaldi/mfcc/[A457][ADAD].39.ark";
  string archive_fn = cmdParser.find("--ark");
  string output_fn  = cmdParser.find("-o");

  int N, dim; float* data; unsigned int* offset;
  loadKaldiArchive(archive_fn, data, offset, N, dim); 

  float** scores = computePairwiseDTW(data, offset, N, dim);

  normalize(scores, N);

  ofstream fout(output_fn);
  range (i, N) {
    range (j, N)
      fout << scores[i][j] << " ";
    fout << endl;
  }
  fout.close();

  return 0;
}

void normalize(float** m, int N) {
  float min = m[0][0];
  float max = m[0][0];

  range (i, N) {
    range (j, N) {
      if (m[i][j] > max) max = m[i][j];
      if (m[i][j] < min) min = m[i][j];
    }
  }
  
  range (i, N)
    range (j, N)
      m[i][j] = (m[i][j] - max) / (max - min) + 1;
}

