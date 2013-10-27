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
vector<string> getRecalledDocId(const vector<string>& retrieved, const vector<string>& answers);
void getSubData(float** sub_data, unsigned int** sub_offset, const float* data, const unsigned int* offset, const vector<size_t>& positions, int N, int dim);
void getSubData(float** sub_data, float** sub_offset, const float* data, const unsigned int* offset, int N, int dim);
float computeDTW(const float* data, const unsigned int* offset, int N, int dim, distance_fn& fn, float eta, int i, int j);
void printSimilarity(Matrix2D<float> m, const vector<string>& docid);
typedef map<string, vector<string> > Answer;
Answer loadAnswer(string filename);
void cutoffWaveFilename(vector<string> &docid);
size_t find(const vector<string>& arr, const string s);

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
  float eta = str2float(cmdParser.find("--eta"));

  Array<string> queries("data/108.query");
  Answer ans = loadAnswer("data/108.ans");

  // map<string, vector<Phone> > phoneLabels;
  // string model_fn = "data/final.mdl";
  // load("data/test.ali.txt"  , model_fn, phoneLabels);
  
  map<string, vector<Phone> > trainPhoneLabels, devPhoneLabels, testPhoneLabels, otherPhoneLabels;
  string model_fn = "data/final.mdl";
  // load("data/train.ali.txt" , model_fn, trainPhoneLabels);
  // load("data/dev.ali.txt"   , model_fn, devPhoneLabels);
  load("data/other.ali.txt" , model_fn, otherPhoneLabels);
  load("data/test.ali.txt"  , model_fn, testPhoneLabels);

  // string query = "COMPACT";
  string query = "ITERATION";
  string archive_fn = "/share/hypothesis/OOV_g2p.kaldi/posterior/" + query + ".76.ark";
  float* data;
  unsigned int* offset;
  int N, dim;
  vector<string> docid;

  loadFeatureArchive(archive_fn, data, offset, N, dim, &docid);
  cutoffWaveFilename(docid);

  /*foreach (i, docid) {
    if (testPhoneLabels.count(docid[i]) == 0)
      continue;

    printf("%s\t", docid[i].c_str());
    const vector<Phone>& labels = testPhoneLabels[docid[i]];
    foreach (j, labels)
      printf("%lu ", labels[j].second);
    printf("\n");
  }*/

  vector<size_t> recall = getRecalls(docid, ans[query]);

  vector<size_t> pos;
  foreach (i, recall) {
    string did = docid[recall[i]];
    if (testPhoneLabels.count(did) != 0 || otherPhoneLabels.count(did) != 0)
      pos.push_back(recall[i]);
    else
      cout << "NOT IN TEST-SET" << endl;
  }

  size_t M = pos.size();

  string root  = "/share/preparsed_files/",
	 set1  = "OOV_g2p_det_add_log_scale",
	 set2  = "OOV_g2p_all_two_no_normalize_-2",
	 s1_fn = root + set1 + "/mul-sim/" + query + ".mul-sim",
	 s2_fn = root + set2 + "/mul-sim/" + query + ".mul-sim";

  Matrix2D<float> s1(s1_fn);
  Matrix2D<float> s2(s2_fn);

  Matrix2D<float> a1(M, M);
  Matrix2D<float> a2(M, M);

  range (i, M) {
    range (j, M) {
      a1[i][j] = s1[pos[i]][pos[j]];
      a2[i][j] = s2[pos[i]][pos[j]];
    }
  }

  vector<string> relDocIds = getRecalledDocId(docid, ans[query]);

  /*cout << s1_fn << endl << endl;
  printSimilarity(a1, relDocIds);
  cout << endl;

  cout << s2_fn << endl << endl;
  printSimilarity(a2, relDocIds);
  cout << endl;*/

  if (pos.size() == 0)
    exit(-1);

  string ustr = docid[pos[0]];
  size_t uindex = find(docid, ustr);

  mylog(ustr);
  mylog(uindex);

  mahalanobis_fn fn(dim);

  computeDTW(data, offset, N, dim, fn, eta, pos[0], pos[1]);
  /*range (i, M) {
    printf("%s vs. %s \n", ustr.c_str(), relDocIds[i].c_str());
    computeDTW(data, offset, N, dim, fn, eta, uindex, pos[i]);
  }
  printf("\n");*/

  /*fn.setDiag(theta_fn);
  range (i, M) {
    float s = computeDTW(data, offset, N, dim, fn, -4, uindex, pos[i]);
    printf("%.5f ", s);
  }
  printf("\n");*/

  delete [] data;
  delete [] offset;

  return 0;
}

float computeDTW(const float* data, const unsigned int* offset, int N, int dim, distance_fn& fn, float eta, int i, int j) {

  size_t rows = (offset[i + 1] - offset[i]) / dim;
  size_t cols = (offset[j + 1] - offset[j]) / dim;

  float* alpha = new float[rows * cols];
  float* beta  = new float[rows * cols];
  float* pdist = new float[rows * cols];

  const float *f1 = data + offset[i];
  const float *f2 = data + offset[j];

  pair_distance(f1, f2, rows, cols, dim, eta, pdist, fn);
  float s = fast_dtw(pdist, rows, cols, dim, eta, alpha, beta);

  printf("rows = %lu, cols = %lu\n", rows, cols);
  range (i, rows) {
    range (j, cols)
      printf("%2.0f ", alpha[i * cols + j] + beta[i * cols + j] - s);
    printf("\n");
  }
  printf("\n");

  delete [] alpha;
  delete [] beta;
  delete [] pdist;

  return s;
}

void printSimilarity(Matrix2D<float> m, const vector<string>& docids) {
  int M = m.getRows();

  range (i, M) {
    printf(BLUE"%s\t"COLOREND, docids[i].c_str());
    range (j, M) {
      if (j < i)
	printf("%.5f ", m[i][j]);
      else
	printf("        ");
    }
    printf("\n");
  }
}

vector<string> getRecalledDocId(const vector<string>& retrieved, const vector<string>& answers) {
  vector<string> docids;

  vector<size_t> pos = getRecalls(retrieved, answers);

  foreach (i, pos)
    docids.push_back(retrieved[pos[i]]);

  return docids;
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


Answer loadAnswer(string filename) {

  FILE* fid = fopen(filename.c_str(), "r");
  size_t qid, dummy;
  char qstring[32], docid[32];
  Answer ans;

  while (fscanf(fid, "%lu %s %s %lu", &qid, qstring, docid, &dummy) != -1)
    ans[qstring].push_back(docid);

  return ans;
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
