#include <string>
#include <iostream>
#include <color.h>
#include <cfloat>

#include <cdtw.h>
// #include <libutility/include/std_common.h>
// #include <libutility/include/thread_util.h>
// #include <libutility/include/utility.h>
// #include <libdtw/include/dtw_parm.h>
// #include <libdtw/include/dtw_util.h>

using std::cout;
using DtwUtil::DtwParm;
using DtwUtil::SegDtwRunner;
using DtwUtil::FrameDtwRunner;
using DtwUtil::SlopeConDtwRunner;
using DtwUtil::FreeFrameDtwRunner;
using namespace std;
using DtwUtil::CumulativeDtwRunner;

float dtw(string f1, string f2, bool printOrNot = true);

int main (int argc, char* argv[]) {

  string q_fname = (argc >= 2) ? string(argv[1]) : "test.mfc";
  string d_fname = (argc >= 3) ? string(argv[2]) : "test.mfc";

  dtw(q_fname, d_fname, false);
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

  return 0;
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
  float maxScore = FLT_MIN;
  for (size_t i = 0; i < nHypo; ++i) {
    float score = hypo_score[i];
    if (printOrNot)
      printf("hypothesized region[%lu]: score = %f, time span = (%d, %d)\n", i, score, hypo_bound[i].first, hypo_bound[i].second);
    //cout << "hypothesized region[" << i << "]: score = " << hypo_score[i] << ", time span = (" << hypo_bound[i].first << ", " << hypo_bound[i].second << ")\n";

    if (score > maxScore)
      maxScore = score;
  }

  return maxScore;
}


// ========== RTK ==========
// Feature f("data/a.fbank");
// f.getHTKHeader().print();
// f.print();
// cout << "size of f.getObsSeq() = " << f.getObsSeq().size() << endl;

