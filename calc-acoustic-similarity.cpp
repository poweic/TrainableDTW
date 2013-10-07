#include <iostream>

#include <cmdparser.h>
#include <array.h>
#include <matrix.h>
#include <util.h>
#include <utility.h>
#include <profile.h>

#include <cdtw.h>

using namespace DtwUtil;
using namespace std;

typedef Matrix2D<double> mat;
void normalize(mat& m, int type = 1);

double cdtw(DtwParm& q_parm, DtwParm& d_parm);
double scdtw(DtwParm& q_parm, DtwParm& d_parm);
double ffdtw(DtwParm& q_parm, DtwParm& d_parm);
double fixdtw(DtwParm& q_parm, DtwParm& d_parm);

template <typename T>
std::pair<T, T> getMinMax(const Matrix2D<T>& m) {
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();

  size_t rows = m.getRows();
  size_t cols = m.getCols();

  for (size_t i=0; i<rows; ++i) {
    for (size_t j=0; j<cols; ++j) {
      T e = m[i][j];
      min = (e < min) ? e : min;
      max = (e > max) ? e : max;
    }
  }

  return std::pair<T, T>(min, max);
}

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .add("-d", "directory containing mfcc files for a certain query")
    .add("-o", "output filename for the acoustic similarity matrix")
    .add("--list", "corresponding list of mfcc files")
    .add("--dtw-type", {"Choose the type of Dynamic Time Warping: ",
			  "fixdtw:\t FixFrameDtwRunner. head-to-head, tail-to-tail",
			  "ffdtw:\t FreeFrameDtwRunner. no head-to-head, tail-to-tail constraint" , 
			  "scdtw:\t SlopeConDtwRunner. Slope-conditioned DTW", 
			  "cdtw:\t CumulativeDtwRunner. Cumulative DTW, considering all paths from head-to-tail."})
    .add("--normalize", "Whether to normalize the acoustic similarity to [0, 1]", false, "true");

  cmdParser
    .addGroup("Distance options")
    .add("--theta", "specify the file containing the diagnol term of Mahalanobis distance (dim=39)", false)
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-4");

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  Profile profile;
  profile.tic();

  // =====================================================
  string path = cmdParser.find("-d") + "/";
  string mat_filename = cmdParser.find("-o");
  string list_filename = cmdParser.find("--list");
  bool normalization = cmdParser.find("--normalize") == "false" ? false : true;
  string theta_filename = cmdParser.find("--theta");
  SMIN::eta = stod(cmdParser.find("--eta"));

  if (!theta_filename.empty()) {
    vector<double> diag(39);
    Array<double> diag2(theta_filename);
    foreach (i, diag)
      diag[i] = diag2[i];
    Bhattacharyya::setDiag(diag);
  }

  enum DTW_TYPE {
    FIXDTW,
    FFDTW,
    SCDTW,
    CDTW
  };

  string typeStr = cmdParser.find("--dtw-type");
  DTW_TYPE type;
  if (typeStr == "fixdtw")
    type = FIXDTW;
  else if (typeStr == "scdtw")
    type = SCDTW;
  else if (typeStr == "cdtw")
    type = CDTW;
  else
    type = FFDTW;

  //vector<string> lists = bash::ls(path);
  Array<string> lists(list_filename);
  foreach (i, lists) {
    
    // Choose Highest number. (i.e. largest granularity)
    // Granularity: word > character > syllable > phone
    for (int j=1; j<50; ++j) {
      string filename = path + lists[i] + "_" + to_string(j) + ".mfc";
      if (exists(filename)) {
	lists[i] = filename;
	break;
      }
    }
  }

  int nSegment = lists.size();

  vector<DtwParm> parms;
  foreach (i, lists)
    parms.push_back(DtwParm(lists[i]));

  mat scores(nSegment, nSegment);

  foreach (i, lists) {
    foreach (j, lists) {
      if (j > i) break;

      double score = 0;
      switch (type) {
	case CDTW:
	  score = cdtw(parms[i], parms[j]);
	  break;
	case FIXDTW:
	  score = fixdtw(parms[i], parms[j]);
	  break;
	case SCDTW:
	  score = scdtw(parms[i], parms[j]);
	  break;
	case FFDTW:
	default:
	  score = ffdtw(parms[i], parms[j]);
	  break;
      }

      scores[i][j] = scores[j][i] = score;
    }
  }

  normalize(scores, 1);
  scores.saveas(mat_filename);
  // =====================================================

  profile.toc();

  return 0;
}

void normalize(mat& m, int type) {
  auto minmax = getMinMax(m);
  auto min = minmax.first;
  auto max = minmax.second;

  switch (type) {
    // [min, max] ==> [min - max, 0] ==> [-1, 0] ==> [0, 1]
    //           shift              scale       shift
    case 1:
      m -= max;
      m /= (max - min);
      m += 1;
      break;
    // [min, max] ==> [min - max, 0] ==> [0, 1]
    //          shift                exp
    case 2:
      m -= max;
      for (size_t i=0; i<m.getRows(); ++i)
	for (size_t j=0; j<m.getCols(); ++j)
	  m[i][j] = exp(m[i][j]);
      
      break;
  }
}

double cdtw(DtwParm& q_parm, DtwParm& d_parm) {
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;

  FrameDtwRunner::nsnippet_ = 10;

  CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(Bhattacharyya::fn);
  dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
  dtwRunner.DTW();

  return dtwRunner.getCumulativeScore();
}

double ffdtw(DtwParm& q_parm, DtwParm& d_parm) {
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;

  FrameDtwRunner::nsnippet_ = 10;
  FreeFrameDtwRunner dtwRunner = FreeFrameDtwRunner(DtwUtil::euclinorm);
  dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
  dtwRunner.DTW();

  float max = numeric_limits<float>::lowest();
  foreach (i, hypo_bound) {
    if (hypo_score[i] > max)
      max = hypo_score[i];
  }
  return (double) max;
}

double fixdtw(DtwParm& q_parm, DtwParm& d_parm) {
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;

  FrameDtwRunner::nsnippet_ = 10;
  FixFrameDtwRunner dtwRunner = FixFrameDtwRunner(DtwUtil::euclinorm);
  dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
  dtwRunner.DTW();

  float max = numeric_limits<float>::lowest();
  foreach (i, hypo_bound) {
    if (hypo_score[i] > max)
      max = hypo_score[i];
  }
  return (double) max;
}

double scdtw(DtwParm& q_parm, DtwParm& d_parm) {
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;

  FrameDtwRunner::nsnippet_ = 10;
  SlopeConDtwRunner dtwRunner = SlopeConDtwRunner(DtwUtil::euclinorm);
  dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
  dtwRunner.DTW();

  float max = numeric_limits<float>::lowest();
  foreach (i, hypo_bound) {
    if (hypo_score[i] > max)
      max = hypo_score[i];
  }
  return (double) max;
}
