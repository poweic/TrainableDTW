#include <iostream>
#include <string>
#include <color.h>
#include <matrix.h>
#include <array.h>
#include <profile.h>

#include <cdtw.h>
#include <archive_io.h>
#include <utility.h>
#include <math_ext.h>
#include <blas.h>
#include <perf.h>

#include <fast_dtw.h>

using namespace std;
typedef Matrix2D<float> mat;

void goAll();
mat fast(const Array<string>& files);

namespace golden {
  //mat go(const Array<string>& files);
  double dtw(string q_fname, string d_fname);
  double dtw(const DtwParm& X, const DtwParm& Y);
};

void run(const vector<string>& phones, const map<size_t, vector<FeatureSeq> >& phoneInstances);

string dir = "data/mfcc/CH_ts/";

int main (int argc, char* argv[]) {

  vector<size_t> perm = randperm(100);
  foreach (i, perm)
    cout << perm[i] << endl;

  return 0;

  // perf::Timer timer;
  // timer.start();

  // string fn = "/media/Data1/hypothesis/SI_word.kaldi/mfcc/[A457][ADAD].39.ark";
  // int M = 500;
  // int N = 500;
  // //float** scores = computePairwiseDTW(fn);
  // timer.stop();

  // printf("Elasped time: %.2f \n", timer.getTime());

  // return 0;

  // mat m(4, 6); //("testing/matrix_lib/A.mat");
  // ext::rand(m);
  // m.print(3);

  // vector<float> v(m.getRows());
  // foreach (i, v)
  //   v[i] = i;

  // ::print(v);

  // mat s = m & v;
  // s.print(3);

  /* vector<size_t> prior(10);
  foreach (i, prior)
    prior[i] = i;

  std::vector<float> pdf(prior.begin(), prior.end());
  float sum = ext::sum(pdf);
  foreach (i, pdf)
    pdf[i] /= sum;

  ::print(pdf);

  vector<size_t> data = ext::sampleDataFrom(pdf, 1000);
  foreach (i, data)
    cout << data[i] << " ";
  cout << endl;*/

  return 0;

  // mat m(1000, 1000);
  // ext::randn(m);

  // int nInf = 0;
  // range (i, m.getRows())
  //   range (j, m.getCols())
  //     assert(!ext::is_inf(m[i][j]));

  //goAll();

  //Array<string> files("test.list");
  //mat s1 = fast(files);
  //mat s2 = golden::go(files);

  //cout << endl << GREEN __DIVIDER__ "Diff" __DIVIDER__ COLOREND << endl;
  //mat s3 = s2 - s1;
  //s3.print();

  return 0;
}

void goAll() {

  string alignmentFile = "data/train.ali.txt";
  string phoneTableFile = "data/phones.txt";
  string modelFile = "data/final.mdl";
  string featArk = "/media/Data1/LectureDSP_script/feat/train.39.ark";

  vector<string> phones = getPhoneMapping(phoneTableFile);

  map<string, vector<Phone> > phoneLabels;
  int nInstance = load(alignmentFile, modelFile, phoneLabels);

  map<size_t, vector<FeatureSeq> > phoneInstances;

  if (featArk.empty())
    return;

  int n = loadFeatureArchive(featArk, phoneLabels, phoneInstances);
  check_equal(n, nInstance);

  run(phones, phoneInstances);
  //size_t nMfccFiles = saveFeatureAsMFCC(phoneInstances, phones);

  cout << "[Done]" << endl;
}

void toDenseFeature(const FeatureSeq& fs, DenseFeature& df) {

  TwoDimVector<float>& data = df.Data();
  data.resize(fs.size(), fs[0].size());

  foreach (i, fs) {
    const DoubleVector& v = fs[i];

    foreach (j, v) 
      data[i][j] = (float) v(j);
  }
}

void run(const vector<string>& phones, const map<size_t, vector<FeatureSeq> >& phoneInstances) {

  DtwParm X;
  DtwParm Y;

  for (auto i=phoneInstances.cbegin(); i != phoneInstances.cend(); ++i) {

    const vector<FeatureSeq>& fSeqs = i->second;

    //ProgressBar pBar("Running DTW...");
    vector<DenseFeature> features(fSeqs.size());
    for (size_t i=0; i<fSeqs.size(); ++i)
      toDenseFeature(fSeqs[i], features[i]);

    for (size_t i=0; i<fSeqs.size(); ++i) {
      //pBar.refresh(double (i+1) / fSeqs.size());

      for (size_t j=0; j<fSeqs.size(); ++j) {

	X.Feat() = features[i];
	Y.Feat() = features[j];

	double score = golden::dtw(X, Y);
      }
    }

  }
}

mat fast(const Array<string>& files) {

  string dummy = dir + "13400.mfc";
  DtwParm q_parm(dummy);
  DtwParm d_parm(dummy);

  int N = files.size();
  mat score(N, N);

  foreach(i, files) {
    foreach(j, files) {
      DtwParm x(dir + files[i]);
      DtwParm y(dir + files[j]);

      q_parm.Feat() = x.Feat();
      d_parm.Feat() = y.Feat();

      score[i][j] = golden::dtw(q_parm, d_parm);
    }
  }

  score.print();
  return score;
}

namespace golden {
  /*mat go(const Array<string>& files) {
    int N = files.size();
    mat score(N, N);

    Profile profile;
    profile.tic();
    int nTimes = 0;

    int nPairs = N*N;
    ProgressBar pBar("Running DTW from mfcc files...");

    foreach(i, files) {
      pBar.refresh(double (i) / N);
      foreach(j, files)
	score[i][j] = dtw(dir + files[i], dir + files[j]);
    }

    double elapsed = profile.toc();
    double avgTime = elapsed / (double) nPairs;
    printf("average time calculating a DTW pair = %.8e, %lu in total\n", avgTime, nPairs);

    return score;
  }*/

  double dtw(const DtwParm& X, const DtwParm& Y) {
    static vector<float> hypo_score;
    static vector<pair<int, int> > hypo_bound;
    hypo_score.clear(); hypo_bound.clear();

    FrameDtwRunner::nsnippet_ = 10;

    CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(Bhattacharyya::fn);
    dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &X, &Y, NULL, NULL);
    dtwRunner.DTW();

    return dtwRunner.getCumulativeScore();

  }

  double dtw(string q_fname, string d_fname) {
    DtwParm q_parm(q_fname);
    DtwParm d_parm(d_fname);

    return dtw(q_parm, d_parm);
  }
};



