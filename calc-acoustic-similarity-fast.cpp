#include <iostream>
#include <fstream>
#include <matrix.h>
#include <color.h>
#include <perf.h>
#include <cmdparser.h>

#include <fast_dtw.h>
using namespace std;

void normalize(float** m, int N);
float calcError(float* s1, float** s2, int N);

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

  // string fn = "/media/Data1/hypothesis/SI_word.kaldi/mfcc/[A457][ADAD].39.ark";
  string archive_fn = cmdParser.find("--ark");
  string output_fn  = cmdParser.find("-o");

  int N, dim; float* data; unsigned int* offset;
  loadKaldiArchive(archive_fn, data, offset, N, dim); 

  perf::Timer timer;

  timer.start();
  float* scores_from_cuda = computePairwiseDTW_in_gpu(data, offset, N, dim);
  timer.stop();
  printf("Elasped time: %.2f secs\n", timer.getTime());

  range (i, N) {
    range (j, N)
      printf("%.4f ", scores_from_cuda[i * N + j]);
    printf("\n");
  }

  timer.reset();
  timer.start();
  float** scores = computePairwiseDTW(data, offset, N, dim);
  timer.stop();
  printf("Elasped time: %.2f secs\n", timer.getTime());

  range (i, N) {
    range (j, N)
      printf("%.4f ", scores[i][j]);
    printf("\n");
  }

  float error = calcError(scores_from_cuda, scores, N);
  mylog(error);

  // normalize(scores, N);
  // 
  // ofstream fout(output_fn.c_str());
  // range (i, N) {
  //   range (j, N)
  //     fout << scores[i][j] << " ";
  //   fout << endl;
  // }
  // fout.close();

  return 0;
}

float calcError(float* s1, float** s2, int N) {
  float error = 0;
  range (i, N)
    range (j, N)
      error += pow(s1[i * N + j] - s2[i][j], 2.0);

  error /= N*N;
  return error;
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

