#include <iostream>
#include <fstream>
#include <color.h>
#include <perf.h>
#include <archive_io.h>
#include <cmdparser.h>

#include <fast_dtw.h>
using namespace std;

void selfTest();
float calcError(float* s1, float* s2, int N);
void normalize(float* m, int N);
void print(float* m, int N);

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .add("--ark", "input feature archive")
    .add("-o", "output filename for the acoustic similarity matrix", false)
    .add("--gpu-enabled", "set to \"true\" to turn on gpu-acceleration", false, "false")
    .add("--self-test", "Perform a self test by calculating the error between GPU & CPU", false, "false");

  cmdParser
    .addGroup("Distance options")
    .add("--theta", "specify the file containing the diagnol term of Mahalanobis distance (dim=39)", false)
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-4");
  
  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  string archive_fn = cmdParser.find("--ark");
  string output_fn  = cmdParser.find("-o");
  bool gpuEnabled   = (cmdParser.find("--gpu-enabled") == "true");
  bool isSelfTest   = (cmdParser.find("--self-test") == "true");

  if (isSelfTest) {
    selfTest();
    return 0;
  }

  int N, dim; float* data; unsigned int* offset;
  loadFeatureArchive(archive_fn, data, offset, N, dim); 

  float* scores;
  if (gpuEnabled)
    scores = computePairwiseDTW_in_gpu(data, offset, N, dim);
  else
    scores = computePairwiseDTW(data, offset, N, dim);

  normalize(scores, N);

  FILE* fid = (output_fn.empty()) ? stdout : fopen(output_fn.c_str(), "w");

  range (i, N) {
    range (j, N)
      fprintf(fid, "%.7f ", scores[i * N + j]);
    fprintf(fid, "\n");
  }
  fclose(fid);

  delete [] scores;

  return 0;
}

void selfTest() {
  string archive_fn = "/media/Data1/hypothesis/SI_word.kaldi/mfcc/[A457][ADAD].39.ark";

  int N, dim; float* data; unsigned int* offset;
  loadFeatureArchive(archive_fn, data, offset, N, dim); 
  mylog(N);

  perf::Timer timer;

  printf(GREEN"===== GPU version ====="COLOREND);
  timer.start();
  float* scores_from_cuda = computePairwiseDTW_in_gpu(data, offset, N, dim);
  timer.stop();
  printf("Elasped time: %.2f secs\n", timer.getTime());

  print(scores_from_cuda, N);


  printf(GREEN"===== CPU version ====="COLOREND);

  timer.reset();
  timer.start();
  float* scores = computePairwiseDTW(data, offset, N, dim);
  timer.stop();
  printf("Elasped time: %.2f secs\n", timer.getTime());

  print(scores, N);


  printf(GREEN"===== Summary ====="COLOREND);
  float error = calcError(scores_from_cuda, scores, N);
  mylog(error);
}

void print(float* m, int N) {
  range (i, N) {
    range (j, N)
      printf("%.6f ", m[i * N + j]);
    printf("\n");
  }
}

float calcError(float* s1, float* s2, int N) {
  float error = 0;
  range (i, N)
    range (j, N)
      error += pow(s1[i * N + j] - s2[i * N + j], 2.0);

  error /= N*N;
  return error;
}

void normalize(float* m, int N) {
  float min = m[0];
  float max = m[0];

  range (i, N) {
    range (j, N) {
      if (m[i * N + j] > max) max = m[i * N + j];
      if (m[i * N + j] < min) min = m[i * N + j];
    }
  }
  
  range (i, N)
    range (j, N)
      m[i * N + j] = (m[i * N + j] - max) / (min - max);
}