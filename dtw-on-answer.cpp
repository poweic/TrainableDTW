#include <iostream>
#include <fstream>
#include <color.h>
#include <perf.h>
#include <archive_io.h>
#include <cmdparser.h>
#include <vector>

#include <array.h>
#include <fast_dtw.h>
using namespace std;

/*void selfTest();
float calcError(float* s1, float* s2, int N);
distance_fn* initDistanceMeasure(string dist_type, size_t dim, string theta_fn);
void normalize(float* m, int N, float eta);
void normalize_in_log(float* m, int N);
void setDiagToOne(float* m, int N);
void setDiagToZero(float* m, int N);*/
void print(float* m, int N);
vector<size_t> getRecalls(const vector<string>& retrieved, const vector<string>& answers);
void getSubData(float** sub_data, unsigned int** sub_offset, const float* data, const unsigned int* offset, const vector<size_t>& positions, int N, int dim);

void getSubData(float** sub_data, float** sub_offset, const float* data, const unsigned int* offset, int N, int dim);

typedef map<string, vector<string> > Answer;
Answer loadAnswer(string filename) {

  FILE* fid = fopen(filename.c_str(), "r");
  size_t qid, dummy;
  char qstring[32], docid[32];
  Answer ans;

  while (fscanf(fid, "%lu %s %s %lu", &qid, qstring, docid, &dummy) != -1)
    ans[qstring].push_back(docid);

  return ans;
}

void cutoffWaveFilename(vector<string> &docid) {
  foreach (i, docid) {
    size_t begin = docid[i].find_last_of('/');
    size_t end = docid[i].find_last_of('_');
    docid[i] = docid[i].substr(begin + 1, end - begin - 1);
  }
}

size_t find(const vector<string>& arr, const string s) {
  foreach (i, arr)
    if (arr[i] == s) return i;
  return -1;
}

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);

  cmdParser
    .addGroup("Distance options")
    .add("--type", "choose \"Euclidean (eu)\", \"Diagonal Manalanobis (ma)\", \"Log Inner Product (lip)\"", false)
    .add("--theta", "specify the file containing the diagnol term of Mahalanobis distance (dim=39)", false)
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-2");

  cmdParser
    .addGroup("Example: ./pair-wise-dtw --ark=data/example.76.ark --type=ma --theta=exp/theta/theta.rand.good");
  
  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  string theta_fn = cmdParser.find("--theta");


  Array<string> queries("data/108.query");
  Answer ans = loadAnswer("data/108.ans");

  // map<string, vector<Phone> > phoneLabels;
  // string model_fn = "data/final.mdl";
  // load("data/test.ali.txt"  , model_fn, phoneLabels);
  
  map<string, vector<Phone> > trainPhoneLabels, devPhoneLabels, testPhoneLabels;
  string model_fn = "data/final.mdl";
  load("data/train.ali.txt" , model_fn, trainPhoneLabels);
  load("data/dev.ali.txt"   , model_fn, devPhoneLabels);
  load("data/test.ali.txt"  , model_fn, testPhoneLabels);

  map<string, vector<Phone> >::iterator i = trainPhoneLabels.begin();
  for (; i != trainPhoneLabels.end(); ++i) {
    printf("%s\t", i->first.c_str());
    foreach (j, i->second)
      printf("%lu ", i->second[j].first);
    printf("\n");
  }

  return 0;
  
  Answer::iterator itr = ans.begin();
  for (; itr != ans.end(); ++itr) {
    string qstring = itr->first;

    string archive_fn = "/share/hypothesis/OOV_g2p.kaldi/posterior/" + qstring + ".76.ark";
    float* data;
    unsigned int* offset;
    int N, dim;
    vector<string> docid;

    loadFeatureArchive(archive_fn, data, offset, N, dim, &docid);
    cutoffWaveFilename(docid);

    mahalanobis_fn fn(dim);
    fn.setDiag(theta_fn);

    foreach (i, docid) {
      if (testPhoneLabels.count(docid[i]) == 0 
	  && devPhoneLabels.count(docid[i]) == 0
	  && trainPhoneLabels.count(docid[i]) == 0) {

	// printf("document %s not in train, dev, test\n", docid[i].c_str());
	continue;
      }

      printf("%s\t", docid[i].c_str());
      const vector<Phone>& labels = testPhoneLabels[docid[i]];
      foreach (j, labels)
	printf("%lu ", labels[j].first);
      printf("\n");

    }
    // vector<size_t> pos = getRecalls(docid, itr->second);

    float* sub_data;
    unsigned int* sub_offset;

    // getSubData(&sub_data, &sub_offset, data, offset, pos, N, dim);

    // float* d = computePairwiseDTW(sub_data, sub_offset, sub_N, dim, fn);
    // print(d, sub_N);
    // delete [] d;

    // delete [] sub_data;
    // delete [] sub_offset;

    delete [] data;
    delete [] offset;
  }

  return 0;
}

vector<size_t> getRecalls(const vector<string>& retrieved, const vector<string>& answers) {
  vector<size_t> positions;
  foreach (i, answers) {
    size_t pos = find(retrieved, answers[i]);
    if (pos != -1)
      positions.push_back(pos);
  }
  return positions;
}

void getSubData(float** sub_data, unsigned int** sub_offset, const float* data, const unsigned int* offset, const vector<size_t>& positions, int N, int dim) {
  size_t totalSize = 0;
  foreach (i, positions) {
    size_t dataSize = offset[positions[i] + 1] - offset[positions[i]];
    totalSize += dataSize;
  }

  int sub_N = positions.size();
  *sub_data = new float[totalSize];
  *sub_offset = new unsigned int[sub_N + 1];

  (*sub_offset)[0] = 0;
  foreach (i, positions) {
    size_t p = positions[i];
    size_t dataSize = offset[p + 1] - offset[p];
    printf("%5lu ", dataSize / dim);
    (*sub_offset)[i + 1] = (*sub_offset)[i] + dataSize;
    memcpy(*sub_data + (*sub_offset)[i], data + offset[p], dataSize * sizeof(float));
  }
  printf("\n\n");
}

  // CmdParser cmdParser(argc, argv);
  // cmdParser
  //   .add("--ark", "input feature archive")
  //   .add("-o", "output filename for the acoustic similarity matrix", false)
  //   .add("--gpu-enabled", "set to \"true\" to turn on gpu-acceleration", false, "false")
  //   .add("--self-test", "Perform a self test by calculating the error between GPU & CPU", false, "false");

  // cmdParser
  //   .addGroup("Distance options")
  //   .add("--scale", "log-scale (log) as distance or linear-scale (linear) as probability density")
  //   .add("--type", "choose \"Euclidean (eu)\", \"Diagonal Manalanobis (ma)\", \"Log Inner Product (lip)\"")
  //   .add("--theta", "specify the file containing the diagnol term of Mahalanobis distance (dim=39)", false)
  //   .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-2");

  // cmdParser
  //   .addGroup("Example: ./pair-wise-dtw --ark=data/example.76.ark --type=ma --theta=exp/theta/theta.rand.good");
  // 
  // if(!cmdParser.isOptionLegal())
  //   cmdParser.showUsageAndExit();

  // string archive_fn = cmdParser.find("--ark");
  // string output_fn  = cmdParser.find("-o");
  // bool gpuEnabled   = (cmdParser.find("--gpu-enabled") == "true");
  // bool isSelfTest   = (cmdParser.find("--self-test") == "true");
  // string theta_fn   = cmdParser.find("--theta");
  // string dist_type  = cmdParser.find("--type");
  // float eta	    = str2float(cmdParser.find("--eta"));
  // string scale	    = cmdParser.find("--scale");

  // if (isSelfTest)
  //   selfTest();

  // perf::Timer timer;
  // timer.start();
  // int N, dim; float* data; unsigned int* offset;
  // loadFeatureArchive(archive_fn, data, offset, N, dim); 

  // mylog(theta_fn);

  // distance_fn* dist = initDistanceMeasure(dist_type, dim, theta_fn);

  // float* scores;
  // if (gpuEnabled)
  //   scores = computePairwiseDTW_in_gpu(data, offset, N, dim);
  // else
  //   scores = computePairwiseDTW(data, offset, N, dim, *dist);

  // setDiagToZero(scores, N);

  // if (scale == "linear")
  //   normalize(scores, N, eta);
  // else if (scale == "log")
  //   normalize_in_log(scores, N);
  // else
  //   exit(-1);

  // FILE* fid = (output_fn.empty()) ? stdout : fopen(output_fn.c_str(), "w");

  // range (i, N) {
  //   range (j, N)
  //     fprintf(fid, "%.7f ", scores[i * N + j]);
  //   fprintf(fid, "\n");
  // }
  // 
  // if (fid != stdout) 
  //   fclose(fid);

  // delete [] scores;

  // timer.elapsed();

  // return 0;

/*distance_fn* initDistanceMeasure(string dist_type, size_t dim, string theta_fn) {
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
*/

void print(float* m, int N) {
  range (i, N) {
    range (j, N)
      printf("%.3f ", m[i * N + j]);
    printf("\n");
  }
}

/*
float calcError(float* s1, float* s2, int N) {
  float error = 0;
  range (i, N)
    range (j, N)
      error += pow(s1[i * N + j] - s2[i * N + j], 2.0);

  error /= N*N;
  return error;
}


void setDiagToOne(float* m, int N) {
  range (i, N)
    m[i * N + i] = 1;
}

void setDiagToZero(float* m, int N) {
  range (i, N)
    m[i * N + i] = 0;
}

void normalize(float* m, int N, float eta) {
  const float MIN_EXP = 12;

  float min = m[0];
  float max = m[0];

  range (i, N) {
    range (j, N) {
      if (m[i * N + j] > max) max = m[i * N + j];
      if (m[i * N + j] < min) min = m[i * N + j];
    }
  }

  if (min - max == 0)
    return;

  float normalizer = MIN_EXP / (max - min) / abs(eta);

  range (i, N*N)
    m[i] = (m[i] - min) * normalizer;

  range (i, N*N)
    m[i] = exp(eta * m[i]);
}

void normalize_in_log(float* m, int N) {

  float min = m[0];
  float max = m[0];

  range (i, N) {
    range (j, N) {
      if (m[i * N + j] > max) max = m[i * N + j];
      if (m[i * N + j] < min) min = m[i * N + j];
    }
  }

  if (min - max == 0)
    return;
  
  range (i, N*N)
    m[i] = (m[i] - max) / (min - max);
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

  euclidean_fn eu;

  timer.reset();
  timer.start();
  float* scores = computePairwiseDTW(data, offset, N, dim, eu);
  timer.stop();
  printf("Elasped time: %.2f secs\n", timer.getTime());

  print(scores, N);


  printf(GREEN"===== Summary ====="COLOREND);
  float error = calcError(scores_from_cuda, scores, N);
  mylog(error);

  exit(1);
}*/
