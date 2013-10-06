#include <iostream>

#include <cmdparser.h>
#include <array.h>
#include <matrix.h>
#include <util.h>
#include <utility.h>
#include <profile.h>
#include <blas.h>

#include <cdtw.h>
#include <corpus.h>

using namespace DtwUtil;
using namespace std;

typedef Matrix2D<double> mat;

double dtw(string f1, string f2, vector<double> *dTheta = NULL);
Array<string> getPhoneList(string filename);
void computeBetweenPhoneDistance(const Array<string>& phones, const string& MFCC_DIR, const size_t N);
mat comparePhoneDistances(const Array<string>& phones);
double average(const mat& m, const double MIN = -3.40e+34);
// int exec(string cmd);
double objective(const mat& scores);
mat statistics (const Array<string>& phones);
void updateTheta(vector<double>& theta, vector<double>& delta);
void deduceCompetitivePhones(const Array<string>& phones, const mat& scores);
void signalHandler(int param);
void saveTheta();
void regSignalHandler();

void validation();
void calcObjective(const vector<tsample>& samples);

void train(size_t batchSize);

string scoreDir;
vector<double> theta;
size_t itr;

int main (int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser
    .addGroup("Generic options")
    .add("-p", "Choose either \"train\" or \"evaluate\".")
    .add("--phone-mapping", "The mapping of phones", false, "data/phones.txt");

  cmdParser
    .addGroup("Distance measure options")
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-64");

  cmdParser
    .addGroup("Training options")
    .add("--batch-size", "number of training samples per batch", false, "10000")
    .add("--resume-training", "resume training using the previous condition", false, "false")
    .add("--mfcc-root", "root directory of MFCC files", false, "data/mfcc/");
  
  cmdParser
    .addGroup("Evaluation options")
    .add("-d", "directory for saving/loading scores", false)
    .add("-o", "filename for scores matrix", false)
    .add("-n", "pick n random instances for each phone when evaulating", false, "100")
    .add("--re-evaluate", "Re-evaluate pair-wise distances for each phones", false, "false");

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  regSignalHandler();

  scoreDir = cmdParser.find("-d") + "/";
  exec("mkdir -p " + scoreDir);

  // Parsering Command Arguments
  string phase = cmdParser.find("-p");

  size_t batchSize = str2int(cmdParser.find("--batch-size"));
  size_t N = str2int(cmdParser.find("-n"));
  string MFCC_DIR = cmdParser.find("--mfcc-root");

  string phones_filename = cmdParser.find("--phone-mapping");
  Array<string> phones = getPhoneList(phones_filename);
  string matFile = cmdParser.find("-o");
  bool resume = cmdParser.find("--resume-training") == "true";
  bool reevaluate = cmdParser.find("--re-evaluate") == "true"; 
  SMIN::eta = stod(cmdParser.find("--eta"));

  theta.resize(39);
  if (resume) {
    Array<double> previous(".theta.restore");
    theta = (vector<double>) previous;
    cout << "Setting theta to previous-trained one" << endl;
  }

  Profile profile;
  profile.tic();

  //validation();
  //saveTheta();
  //return 0;

  if (phase == "train")
    train(batchSize);
  else if (phase == "evaluate") {

    if (reevaluate)
      computeBetweenPhoneDistance(phones, MFCC_DIR, N);

    mat scores = comparePhoneDistances(phones);

    if (!matFile.empty())
      scores.saveas(matFile);

    deduceCompetitivePhones(phones, scores);

    debug(objective(scores));
  }

  profile.toc();

  saveTheta();
  return 0;
}

void validation() {
  Corpus corpus("data/phones.txt");
  const size_t MINIATURE_SIZE = 1000;
  vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

  cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

  theta.resize(39);
  fillwith(theta, 1);

  for (itr=0; itr<10000; ++itr) {

    auto t = theta;

    // TODO use 38-dim !! 'Cause of the constraint: u1 + u2 + .. + u39 = 1
    vector<double> dTheta(39);
    foreach (i, samples) {
      vector<double> dThetaPerSample(39);
      auto cscore = dtw(samples[i].first.first, samples[i].first.second, &dThetaPerSample);
      bool positive = samples[i].second;
      dTheta = positive ? (dTheta + dThetaPerSample) : (dTheta - dThetaPerSample);
    }

    dTheta = dTheta / (double) samples.size();
    updateTheta(theta, dTheta);
    double diff = norm(theta - t);
    debug(diff);
    // printf("#"BLUE"%5lu:"COLOREND"\tdiff = "GREEN "%.6e" COLOREND ":\t", itr, diff);
    //print(theta);

    calcObjective(samples);
  }
  
}

void calcObjective(const vector<tsample>& samples) {

  double obj = 0;
  foreach (i, samples) {
    auto cscore = dtw(samples[i].first.first, samples[i].first.second);
    bool positive = samples[i].second;
    obj += (positive) ? cscore : (-cscore);
  }

  cout << "objective = " << GREEN << obj << COLOREND << endl;
  //cin.get();
}

void train(size_t batchSize) {
  Corpus corpus("data/phones.txt");

  if (!corpus.isBatchSizeApprop(batchSize))
    return;

  const size_t epoch = 1;
  size_t nBatch = corpus.size() / batchSize;
  cout << "# of batches = " << nBatch << endl;
  for (itr=0; itr<nBatch; ++itr) {
    vector<tsample> samples = corpus.getSamples(batchSize);
    cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

    auto t = theta;

    // TODO use 38-dim !! 'Cause of the constraint: u1 + u2 + .. + u39 = 1
    vector<double> dTheta(39);
    size_t nPositive = 0;
    size_t nNegative = 0;
    foreach (i, samples) {
      vector<double> dThetaPerSample(39);
      auto cscore = dtw(samples[i].first.first, samples[i].first.second, &dThetaPerSample);

      bool positive = samples[i].second;
      dTheta = positive ? (dTheta + dThetaPerSample) : (dTheta - dThetaPerSample);

      if (positive)
	++nPositive;
      else
	++nNegative;
    }

    dTheta = dTheta / (double) samples.size();
    updateTheta(theta, dTheta);

    //double diff = norm(theta - t);
    //printf(GREEN "%.6e" COLOREND ":\t", diff);
    //printf(" # of positive = %lu, # of negative = %lu\n", nPositive, nNegative);
    print(theta);
    saveTheta();
  }
}

void deduceCompetitivePhones(const Array<string>& phones, const mat& scores) {
  for (size_t i=0; i<scores.getRows(); ++i) {
    double avg = 0;
    for (size_t j=0; j<scores.getCols(); ++j)
      avg += scores[j][i];
    avg /= phones.size() - 2;
    printf("#%2lu phone" GREEN "( %7s )" COLOREND ": within-phone score = %.4f, avg score between other phones = %.4f", i, phones[i].c_str(), scores[i][i], avg);

    double diff = avg - scores[i][i];
    printf(", diff = "ORANGE"%s"COLOREND"%.4f\n", (sign(diff) > 0 ? "+" : "-"), abs(diff));
  }
  
  int nCompetingPair = 0;
  for (size_t i=2; i<scores.getRows(); ++i) {
    printf("%-10s: [", phones[i].c_str());
    int counter = 0;
    for (size_t j=2; j<scores.getCols(); ++j) {
      if (scores[j][i] > scores[i][i]) {
	++nCompetingPair;
	++counter;
	printf("%s ", phones[j].c_str()); 
      }
    }
    printf("] => %d\n\n", counter);
  }
  nCompetingPair /= 2;
  printf("# of competing phone pairs = %d\n", nCompetingPair);
}

double objective(const mat& scores) {
  double obj = 0;
  for (size_t i=0; i<scores.getRows(); ++i) {
    obj += scores[i][i];
    for (size_t j=0; j<i; ++j)
      obj -= scores[i][j];
  }
  return obj;
}

mat comparePhoneDistances(const Array<string>& phones) {

  mat scores(phones.size(), phones.size());

  foreach (i, phones) {
    if (i <= 1) continue;
    string phone1 = phones[i];

    foreach (j, phones) {
      if (j <= 1) continue;
      if (j > i) break;
      string phone2 = phones[j];

      string file = scoreDir + int2str(i) + "-" + int2str(j) + ".mat";
      double avg = average(mat(file));

      //printf("avg = %.4f between phone #%d : %6s and phone #%d : %6s", avg, i, phone1.c_str(), j, phone2.c_str());
      scores[i][j] = scores[j][i] = avg;
    }
  }

  return scores;
}

mat statistics(const Array<string>& phones) {
  vector<Array<string> > lists(phones.size());

  // Run statistics of nPairs to get the distribution
  double nPairsInTotal = 0;
  mat nPairs(phones.size(), phones.size());

  // Load all lists for 74 phones
  foreach (i, lists) {

    cout << "data/train/list/" + int2str(i) + ".list" << endl;

    lists[i] = Array<string>("data/train/list/" + int2str(i) + ".list");
  }

  // Compute # of pairs
  foreach (i, phones) {
    if (i <= 1) continue;
    size_t M = lists[i].size();

    foreach (j, phones) {
      if (j <= 1) continue;
      if (j >= i) break;

      size_t N = lists[j].size();
      nPairs[i][j] = nPairs[j][i] = M * N;
    }
    nPairs[i][i] = M * (M-1) / 2;
  }

  // Compute # of total pairs
  foreach (i, phones) {
    foreach (j, phones) {
      if (j > i) break;
      nPairsInTotal += nPairs[i][j];
    }
  }

  debug(nPairsInTotal);

  // Normalize
  nPairs *= 1.0 / nPairsInTotal;
  return nPairs;
}

void computeBetweenPhoneDistance(const Array<string>& phones, const string& MFCC_DIR, const size_t N) {
  vector<Array<string> > lists(phones.size());

  const size_t MAX_STATIONARY_ITR = 1000;
  size_t nItrStationary = 0;

  foreach (i, lists)
    lists[i] = Array<string>("data/train/list/" + int2str(i) + ".list");

  foreach (i, phones) {
    if (i <= 1) continue;
    string phone1 = phones[i];

    foreach (j, phones) {
      if (j <= 1) continue;
      if (j > i) break;

      string phone2 = phones[j];

      printf("Computing distances between phone #%2lu (%8s) & phone #%2lu (%8s)\n", i, phone1.c_str(), j, phone2.c_str());
      
      int rows = MIN(lists[i].size(), N);
      int cols = MIN(lists[j].size(), N);

      mat score(rows, cols);
    
      foreach (m, lists[i]) {
	if (m >= N) break;
	foreach (n, lists[j]) {
	  if (n >= N) break;

	  string f1 = MFCC_DIR + phone1 + "/" + lists[i][m];
	  string f2 = MFCC_DIR + phone2 + "/" + lists[j][n];
	  score[m][n] = dtw(f1, f2);
	}
      }

      string file = scoreDir + int2str(i) + "-" + int2str(j) + ".mat";
      score.saveas(file);
    }
  }

}

void updateTheta(vector<double>& theta, vector<double>& delta) {
  static double learning_rate = 0.0000001;
  double maxNorm = 1;

  // Adjust Learning Rate || Adjust delta 
  // TODO go Google some algorithms to adjust learning_rate, such as Line Search, and of course some packages
  /*if (norm(delta) * learning_rate > maxNorm) {
    //delta = delta / norm(delta) * maxNorm / learning_rate;
    learning_rate = maxNorm / norm(delta);
  }*/

  debug(norm(delta));
  debug(learning_rate);
  foreach (i, theta)
    theta[i] -= learning_rate*delta[i];

  // Enforce every diagonal element >= 0

  // theta = theta / vecsum(theta); <== This is kind of erroneous
  // , 'cause this isn't the way updating theta ( according to Gradient Descent)
  // , and therefore the objective function is not gauranteed to be monotonically Inc/Dec
  //theta = vmax(0, theta);

  Bhattacharyya::setDiag(theta);
}

double dtw(string q_fname, string d_fname, vector<double> *dTheta) {
  static vector<float> hypo_score;
  static vector<pair<int, int> > hypo_bound;
  hypo_score.clear(); hypo_bound.clear();

  DtwParm q_parm(q_fname);
  DtwParm d_parm(d_fname);
  FrameDtwRunner::nsnippet_ = 10;

  //FrameDtwRunner *dtwRunner;
  //dtwRunner = new FreeFrameDtwRunner(DtwUtil::euclinorm);
  //dtwRunner = new SlopeConDtwRunner(DtwUtil::euclinorm);
  CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(Bhattacharyya::fn);
  dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
  dtwRunner.DTW();

  //vector<double> ddTheta = ;
  if (dTheta != NULL)
    *dTheta = calcDeltaTheta(&dtwRunner);

  return dtwRunner.getCumulativeScore();
}

/*int exec(string cmd) {
  return system(cmd.c_str());
}*/

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

double average(const mat& m, const double MIN) {
  double total = 0;
  int counter = 0;

  for (size_t i=0; i<m.getRows(); ++i) {
    for (size_t j=0; j<m.getCols(); ++j) {
      if (m[i][j] < MIN)
	continue;

      total += m[i][j];
      ++counter;
    }
  }
  return total / counter;
}

void saveTheta() {
  Array<double> t(theta.size());
  foreach (i, t)
    t[i] = theta[i];
  t.saveas(".theta.restore");
}

void signalHandler(int param) {
  cout << RED "[Interrupted]" COLOREND << " aborted by user. # of iteration = " << itr << endl;
  cout << ORANGE "[Logging]" COLOREND << " saving configuration and experimental results..." << endl;

  saveTheta();
  cout << GREEN "[Done]" COLOREND << endl;

  exit(-1);
}


void regSignalHandler () {
  if (signal (SIGINT, signalHandler) == SIG_ERR)
    cerr << "Cannot catch signal" << endl;
}

