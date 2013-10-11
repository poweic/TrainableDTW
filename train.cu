#include <iostream>

#include <cmdparser.h>
#include <array.h>
#include <matrix.h>
#include <util.h>
#include <utility.h>
#include <profile.h>
#include <blas.h>
#include <cdtw.h>

#include <dnn.h>
#include <trainable_dtw.h>
#include <phone_stat.h>
#include <corpus.h>

using namespace DtwUtil;
using namespace std;

void initModel(Model& model, size_t nLayer, size_t nHiddenNodes, float lr);
void evaluate(bool reevaluate, const Array<string>& phones, string MFCC_DIR, size_t N, string matFile);
Array<string> getPhoneList(string filename);
void signalHandler(int param);
void regSignalHandler();

string scoreDir;
vector<double> theta;
size_t itr;
Model model;

int main (int argc, char* argv[]) {
  
  CmdParser cmdParser(argc, argv);
  cmdParser
    .addGroup("Generic options")
    .add("-p", "Choose either \"validate\", \"train\" or \"evaluate\".")
    .add("--phone-mapping", "The mapping of phones", false, "data/phones.txt");

  cmdParser
    .addGroup("Distance measure options")
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-64");

  cmdParser
    .addGroup("Training options")
    .add("--batch-size", "number of training samples per batch", false, "10000")
    .add("--resume-training", "resume training using the previous condition", false, "false")
    .add("--mfcc-root", "root directory of MFCC files", false, "data/mfcc/")
    .add("--learning-rate", "learning rate", false, "-0.0001");

  cmdParser
    .addGroup("Deep Neural Network options:")
    .add("--layers", "Number of hidden layer in both PP and DTW", false, "3")
    .add("--hidden-nodes", "Number of hidden nodes per hidden layer", false, "64");
  
  cmdParser
    .addGroup("Evaluation options")
    .add("-d", "directory for saving/loading scores", false)
    .add("-o", "filename for scores matrix", false)
    .add("-n", "pick n random instances for each phone when evaulating", false, "100")
    .add("--re-evaluate", "Re-evaluate pair-wise distances for each phones", false, "false");

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  //regSignalHandler();

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
  SMIN::eta = str2double(cmdParser.find("--eta"));
  bool validationOnly = cmdParser.find("--validation-only") == "true";
  float lr = str2float(cmdParser.find("--learning-rate"));
  size_t nHiddenLayer = str2int(cmdParser.find("--layers"));
  size_t nHiddenNodes = str2int(cmdParser.find("--hidden-nodes"));

  theta.resize(39);
  if (resume) {
    Array<double> previous(".theta.restore");
    theta = (vector<double>) previous;
    cout << "Setting theta to previous-trained one" << endl;
  }

  Profile profile;
  profile.tic();

  //model.load("data/dtwdnn.model/");
  //initModel(model, nHiddenLayer, nHiddenNodes, lr);

  if (phase == "validate") {
    // dtwdnn::validation();
    dtwdiag39::validation();
  }
  else if (phase == "train") {
    dtwdiag39::train(batchSize);
    // dtwdnn::train(batchSize);
  }
  else if (phase == "evaluate") {

  }

  profile.toc();

  dtwdiag39::saveTheta();
  return 0;
}

void initModel(Model& model, size_t nLayer, size_t nHiddenNodes, float lr) {
  vector<size_t> d1(nLayer + 2), d2(nLayer + 2);
  printf("# of hidden layer = %lu, # of node per hidden layer = %lu\n", nLayer, nHiddenNodes);

  d1[0] = 39; d1.back() = 74;
  d2[0] = 74; d2.back() = 1;

  for (size_t i=1; i<d1.size() - 1; ++i)
    d1[i] = d2[i] = nHiddenNodes;

  model = Model(d1, d2);
  model.setLearningRate(lr);
}

void evaluate(bool reevaluate, const Array<string>& phones, string MFCC_DIR, size_t N, string matFile) {
  if (reevaluate)
    computeBetweenPhoneDistance(phones, MFCC_DIR, N, scoreDir);

  Matrix2D<double> scores = comparePhoneDistances(phones, scoreDir);

  if (!matFile.empty())
    scores.saveas(matFile);

  deduceCompetitivePhones(phones, scores);

  debug(objective(scores));
}

Array<string> getPhoneList(string filename) {

  Array<string> list;

  fstream file(filename.c_str());
  string line;
  while( std::getline(file, line) ) {
    vector<string> sub = split(line, ' ');
    string phone = sub[0];
    list.push_back(phone);
  }
  file.close();

  return list;
}

void signalHandler(int param) {
  cout << RED "[Interrupted]" COLOREND << " aborted by user. # of iteration = " << itr << endl;
  cout << ORANGE "[Logging]" COLOREND << " saving configuration and experimental results..." << endl;

  dtwdiag39::saveTheta();
  cout << GREEN "[Done]" COLOREND << endl;

  exit(-1);
}


void regSignalHandler () {
  if (signal (SIGINT, signalHandler) == SIG_ERR)
    cerr << "Cannot catch signal" << endl;
}
