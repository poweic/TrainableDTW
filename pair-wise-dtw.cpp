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
distance_fn* initDistanceMeasure(string dist_type, size_t dim, string theta_fn);
// void normalize(float* m, int N, float eta);
// void normalize_in_log(float* m, int N);
void cvtDistanceToSimilarity(float* m, int N);
void print(FILE* fid, float* m, int N);

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .add("--ark", "input feature archive")
    .add("-o", "output filename for the acoustic similarity matrix", false)
    .add("--gpu-enabled", "set to \"true\" to turn on gpu-acceleration", false, "false")
    .add("--self-test", "Perform a self test by calculating the error between GPU & CPU", false, "false");

  cmdParser
    .addGroup("Distance options")
    .add("--type", "choose \"Euclidean (eu)\", \"Diagonal Manalanobis (ma)\", \"Log Inner Product (lip)\"")
    .add("--theta", "specify the file containing the diagnol term of Mahalanobis distance (dim=39)", false)
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-2");

  cmdParser
    .addGroup("Example: ./pair-wise-dtw --ark=data/example.76.ark --type=ma --theta=exp/theta/theta.rand.good");
  
  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  string archive_fn = cmdParser.find("--ark");
  string output_fn  = cmdParser.find("-o");
  bool gpuEnabled   = (cmdParser.find("--gpu-enabled") == "true");
  bool isSelfTest   = (cmdParser.find("--self-test") == "true");
  string theta_fn   = cmdParser.find("--theta");
  string dist_type  = cmdParser.find("--type");
  float eta	    = str2float(cmdParser.find("--eta"));

  if (isSelfTest)
    selfTest();

  perf::Timer timer;
  timer.start();
  int N, dim; float* data; unsigned int* offset;
  loadFeatureArchive(archive_fn, data, offset, N, dim); 

  mylog(theta_fn);

  distance_fn* dist = initDistanceMeasure(dist_type, dim, theta_fn);

  float* scores = NULL;
  if (gpuEnabled) {
#ifdef __CUDACC__
    scores = computePairwiseDTW_in_gpu(data, offset, N, dim);
#else
    printf("GPU-version of Dynamic Time Warping is not supported. Please re-compile\n");
#endif
  }
  else {
    scores = computePairwiseDTW(data, offset, N, dim, *dist, eta);
  }

  cvtDistanceToSimilarity(scores, N);

  FILE* fid = (output_fn.empty()) ? stdout : fopen(output_fn.c_str(), "w");
  print(fid, scores, N);
  if (fid != stdout) 
    fclose(fid);

  delete [] scores;

  timer.elapsed();

  return 0;
}

distance_fn* initDistanceMeasure(string dist_type, size_t dim, string theta_fn) {
  distance_fn* dist;

  if (dist_type == "ma") {
    dist = new mahalanobis_fn(dim);
    dynamic_cast<mahalanobis_fn*>(dist)->setDiag(theta_fn);
  }
  else if (dist_type == "lip") {
    dist = new log_inner_product_fn(dim);
    dynamic_cast<mahalanobis_fn*>(dist)->setDiag(theta_fn);
  }
  else if (dist_type == "eu")
    dist = new euclidean_fn;
  else {
    fprintf(stderr, "--type unspecified or unknown\n");
    exit(-1);
  }

  return dist;
}

void print(FILE* fid, float* m, int N) {
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j)
      fprintf(fid, "%.6f ", m[i * N + j]);
    fprintf(fid, "\n");
  }
}

float calcError(float* s1, float* s2, int N) {
  float error = 0;
  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      error += pow(s1[i * N + j] - s2[i * N + j], 2.0);

  error /= N*N;
  return error;
}

void normalize(float* m, int N, float eta) {
  const float MIN_EXP = 12;

  float min = m[0];
  float max = m[0];

  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      if (m[i * N + j] > max) max = m[i * N + j];
      if (m[i * N + j] < min) min = m[i * N + j];
    }
  }

  if (min - max == 0)
    return;

  float normalizer = MIN_EXP / (max - min) / abs(eta);

  for (int i=0; i<N*N; ++i)
    m[i] = (m[i] - min) * normalizer;

  for (int i=0; i<N*N; ++i)
    m[i] = exp(eta * m[i]);
}

void cvtDistanceToSimilarity(float* m, int N) {
  float min = m[0];
  float max = m[0];

  for (int i=0; i<N; ++i) {
    for (int j=0; j<i; ++j) {
      if (m[i * N + j] > max) max = m[i * N + j];
      if (m[i * N + j] < min) min = m[i * N + j];
    }
  }

  printf("max = %.7f, min = %.7f \n", max, min);

  if (min - max == 0)
    return;
  
  for (int i=0; i<N; ++i)
    m[i*N + i] = min;
  
  for (int i=0; i<N*N; ++i)
    m[i] = abs((m[i] - max) / (min - max));
}

void normalize_in_log(float* m, int N) {

  float min = m[0];
  float max = m[0];

  for (int i=0; i<N; ++i) {
    for (int j=0; j<i; ++j) {
      if (m[i * N + j] > max) max = m[i * N + j];
      if (m[i * N + j] < min) min = m[i * N + j];
    }
  }

  printf("max = %.7f, min = %.7f \n", max, min);

  if (min - max == 0)
    return;
  
  for (int i=0; i<N; ++i)
    m[i*N + i] = min;
  
  for (int i=0; i<N*N; ++i)
    m[i] = abs((m[i] - max) / (min - max));
}

void selfTest() {

#ifdef __CUDACC__
  string archive_fn = "/media/Data1/hypothesis/SI_word.kaldi/mfcc/[A457][ADAD].39.ark";

  int N, dim; float* data; unsigned int* offset;
  // loadFeatureArchive(archive_fn, data, offset, N, dim); 
  mylog(N);

  perf::Timer timer;

  printf(GREEN"===== GPU version ====="COLOREND);
  timer.start();
  float* scores_from_cuda = computePairwiseDTW_in_gpu(data, offset, N, dim);
  timer.stop();
  printf("Elasped time: %.2f secs\n", timer.getTime());

  print(stdout, scores_from_cuda, N);


  printf(GREEN"===== CPU version ====="COLOREND);

  euclidean_fn eu;

  timer.reset();
  timer.start();
  float* scores = computePairwiseDTW(data, offset, N, dim, eu, -4);
  timer.stop();
  printf("Elasped time: %.2f secs\n", timer.getTime());

  print(stdout, scores, N);


  printf(GREEN"===== Summary ====="COLOREND);
  float error = calcError(scores_from_cuda, scores, N);
  mylog(error);
#endif

  exit(1);
}
