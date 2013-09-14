#include <string>
#include <iostream>
#include <limits>

#include <color.h>
#include <cmdparser.h>
#include <array.h>
#include <matrix.h>
#include <util.h>
#include <utility.h>

#include <cdtw.h>

using std::cout;
using DtwUtil::DtwParm;
using DtwUtil::SegDtwRunner;
using DtwUtil::FrameDtwRunner;
using DtwUtil::SlopeConDtwRunner;
using DtwUtil::FreeFrameDtwRunner;
using namespace std;
using DtwUtil::CumulativeDtwRunner;

#define FLOAT_MIN (std::numeric_limits<float>::lowest())
#define foreach(i, arr) for (size_t i=0; i<arr.size(); ++i)

float dtw(string f1, string f2, bool printOrNot = true);
Array<string> getPhoneList(string filename);
void computeBetweenPhoneDistance(const Array<string>& phones, const size_t N = 100);
void computeWithinPhoneDistance(const Array<string>& phones, const size_t N = 100);
Matrix2D<double> comparePhoneDistances(const Array<string>& phones);
double average(const Matrix2D<double>& m);

string listDir("data/train/list/");
string mfccDir("data/mfcc/");
string scoreDir("data/score/");

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser.regOpt("--print", "print detail (default=false)", false);

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  bool printOrNot = (cmdParser.find("--print") == "true") ? true : false;

  const size_t N = 100;
  Array<string> phones = getPhoneList("data/phones.txt");

  //computeBetweenPhoneDistance(phones);
  //computeWithinPhoneDistance(phones);

  Matrix2D<double> scores = comparePhoneDistances(phones);
  
  for (size_t i=0; i<scores.getRows(); ++i) {
    double avg = 0;
    for (size_t j=0; j<scores.getCols(); ++j)
      avg += scores[j][i];
    avg /= phones.size() - 2;
    printf("#%2lu phone" GREEN "(%8s)" COLOREND ": within-phone score = %.4f, avg score between other phones = %.4f\n", i, phones[i].c_str(), scores[i][i], avg);
  }
  
  int nCompetingPair = 0;
  for (size_t i=2; i<scores.getRows(); ++i) {
    for (size_t j=2; j<i; ++j) {
      if (scores[j][i] > scores[i][i]) {
	++nCompetingPair;
	printf("phone #%2lu (%5s) and phone #%2lu (%5s) are competing !\n", i, phones[i].c_str(), j, phones[j].c_str()); 
      }
    }
  }

  printf("# of competing phone pairs = %d\n", nCompetingPair);

  return 0;
}

Matrix2D<double> comparePhoneDistances(const Array<string>& phones) {

  Matrix2D<double> scores(phones.size(), phones.size());

  foreach (i, phones) {
    if (i <= 1) continue;
    string phone1 = phones[i];

    string file = scoreDir + int2str(i) + ".mat";
    double inPhoneDistance = average(Matrix2D<double>(file));
    scores[i][i] = inPhoneDistance;

    foreach (j, phones) {
      if (j <= 1) continue;
      if (j >= i) break;
      string phone2 = phones[j];

      string file = scoreDir + int2str(i) + "-" + int2str(j) + ".mat";
      double avg = average(Matrix2D<double>(file));

      //printf("avg = %.4f between phone #%d : %6s and phone #%d : %6s", avg, i, phone1.c_str(), j, phone2.c_str());
      scores[i][j] = scores[j][i] = avg;
    }
  }

  return scores;
}

double average(const Matrix2D<double>& m) {
  double total = 0;
  int counter = 0;

  for (size_t i=0; i<m.getRows(); ++i) {
    for (size_t j=0; j<m.getCols(); ++j) {
      if (m[i][j] > -3.40e+34) {
	total += m[i][j];
	++counter;
      }
    }
  }
  return total / counter;
}

void computeBetweenPhoneDistance(const Array<string>& phones, const size_t N) {
  vector<Array<string> > lists(phones.size());
  foreach (i, lists) {
    string listFilename = listDir + int2str(i) + ".list";
    lists[i] = Array<string>(listFilename);
  }

  foreach (i, phones) {
    if (i <= 1) continue;
    string phone1 = phones[i];

    foreach (j, phones) {
      if (j <= 1) continue;
      if (j >= i) break;

      string phone2 = phones[j];

      printf("Computing distances between %s & %s\n", phone1.c_str(), phone2.c_str());
      int rows = MIN(lists[i].size(), N);
      int cols = MIN(lists[j].size(), N);

      Matrix2D<double> score(rows, cols);
    
      foreach (m, lists[i]) {
	if (m >= N) break;
	foreach (n, lists[j]) {
	  if (n >= N) break;

	  string f1 = mfccDir + phone1 + "/" + lists[i][m];
	  string f2 = mfccDir + phone2 + "/" + lists[j][n];
	  
	  score[m][n] = dtw(f1, f2, false);
	  //cout << f1 << " : " << f2 << endl;
	}
      }

      string file = scoreDir + int2str(i) + "-" + int2str(j) + ".mat";
      score.saveas(file);

    }
  }
}

void computeWithinPhoneDistance(const Array<string>& phones, const size_t N) {

  foreach (i, phones) {
    string phone = phones[i];
    string listFilename = listDir + int2str(i) + ".list";
    Array<string> list(listFilename);

    int n = MIN(N, list.size());
    Matrix2D<double> score(n, n);

    cout << phone << endl;
    if (i <= 1)
      continue;

    foreach (j, list) {
      if (j >= N) break;
      foreach (k, list) {
	if (k >= N) break;
	string f1 = mfccDir + phone + "/" + list[j];
	string f2 = mfccDir + phone + "/" + list[k];
	//cout << f1 << " : " << f2 << endl;
	score[j][k] = dtw(f1, f2, false);
      }
    }

    score.saveas(scoreDir + int2str(i) + ".mat");
  }
  //float score = dtw(q_fname, d_fname, printOrNot);
  //printf("%.5f\n", score);
}

float dtw(string q_fname, string d_fname, bool printOrNot) {
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;
  DtwParm q_parm, d_parm;

  hypo_score.clear(); hypo_bound.clear();
  q_parm.LoadParm(q_fname);
  d_parm.LoadParm(d_fname);
  SlopeConDtwRunner scdtw_runner(DtwUtil::euclinorm);
  FrameDtwRunner::nsnippet_ = 10;
  scdtw_runner.InitDtw(&hypo_score,
                       &hypo_bound, /* (start, end) frame */
                       NULL, /* do not backtracking */
                       &q_parm,
                       &d_parm,
                       NULL, /* full time span */
                       NULL); /* full time span */
  scdtw_runner.DTW();

  if (printOrNot)
    cout << BLUE "----- Slope-Constrained DTW -----" COLOREND << endl;;

  size_t nHypo = hypo_score.size();
  float maxScore = FLOAT_MIN;
  for (size_t i = 0; i < nHypo; ++i) {
    float score = hypo_score[i];
    if (printOrNot)
      printf("hypothesized region[%lu]: score = %f, time span = (%d, %d)\n", i, score, hypo_bound[i].first, hypo_bound[i].second);
    //cout << "hypothesized region[" << i << "]: score = " << hypo_score[i] << ", time span = (" << hypo_bound[i].first << ", " << hypo_bound[i].second << ")\n";

    if (score > maxScore)
      maxScore = score;
  }

  return maxScore;

  /*
  // Cumulative-based DTW //
  hypo_score.clear(); hypo_bound.clear();
  q_parm.LoadParm(q_fname);
  d_parm.LoadParm(d_fname);
  CumulativeDtwRunner cdtw_runner(DtwUtil::euclinorm);
  FrameDtwRunner::nsnippet_ = 10;
  cdtw_runner.InitDtw(&hypo_score,
                       &hypo_bound, // (start, end) frame 
                       NULL, // do not backtracking 
                       &q_parm,
                       &d_parm,
                       NULL, // full time span
                       NULL); // full time span

  cout << "\33[34m-- Cumulative DTW --\33[0m\n";
  cdtw_runner.DTW();
  */
}

Array<string> getPhoneList(string filename) {

  Array<string> list;

  fstream file(filename);
  string line;
  while( std::getline(file, line) ) {
    vector<string> sub = split(line, ' ');
    string phone = sub[0];
    list.push_back(phone);
  }
  file.close();

  return list;
}

// ========== RTK ==========
// Feature f("data/a.fbank");
// f.getHTKHeader().print();
// f.print();
// cout << "size of f.getObsSeq() = " << f.getObsSeq().size() << endl;

