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

void initModel(Model& model, size_t feat_dim, size_t nLayer, size_t nHiddenNodes, float lr);
void evaluate(bool reevaluate, const Array<string>& phones, string MFCC_DIR, size_t N, string matFile);
Array<string> getPhoneList(string filename);
void signalHandler(int param);
void regSignalHandler();

string scoreDir;
Model model;
vector<double> theta;

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
    .add("--model", "choose a distance model, either \"dnn\" or \"diag\"", false, "dnn")
    .add("--batch-size", "number of training samples per batch", false, "1000")
    .add("--resume-training", "resume training using the previous condition", false, "false")
    .add("--learning-rate", "learning rate", false, "-0.0001")
    .add("--theta-output", "choose a file to save theta", false, ".theta.restore");

  cmdParser
    .addGroup("Training Corpus options:")
    .add("--feat-dim", "dimension of feature vector (ex: 39 for mfcc)", false, "39")
    .add("--mfcc-root", "root directory of MFCC files", false, "data/mfcc/");

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

  size_t batchSize  = str2int(cmdParser.find("--batch-size"));
  size_t N	    = str2int(cmdParser.find("-n"));
  string MFCC_DIR   = cmdParser.find("--mfcc-root");

  SubCorpus::setMfccDirectory(MFCC_DIR);

  string phones_filename  = cmdParser.find("--phone-mapping");
  Array<string> phones	  = getPhoneList(phones_filename);
  string matFile	  = cmdParser.find("-o");
  bool resume		  = cmdParser.find("--resume-training") == "true";
  bool reevaluate	  = cmdParser.find("--re-evaluate") == "true"; 
  SMIN::eta		  = str2double(cmdParser.find("--eta"));
  bool validationOnly	  = cmdParser.find("--validation-only") == "true";
  float lr		  = str2float(cmdParser.find("--learning-rate"));
  size_t nHiddenLayer	  = str2int(cmdParser.find("--layers"));
  size_t nHiddenNodes	  = str2int(cmdParser.find("--hidden-nodes"));
  string m		  = cmdParser.find("--model");
  string thetaFilename	  = cmdParser.find("--theta-output");

  size_t feat_dim	  = str2int(cmdParser.find("--feat-dim"));

  Profile profile;
  profile.tic();

  if (m == "dnn") {
    dtwdnn::initModel(model, feat_dim, nHiddenLayer, nHiddenNodes, lr);

    if (phase == "validate")
      dtwdnn::validation();
    else if (phase == "train")
      dtwdnn::train(batchSize);
  }
  else if (m == "diag") {

    dtwdiag::initModel(resume, feat_dim);

    if (phase == "validate")
      dtwdiag::validation();
    else if (phase == "train")
      dtwdiag::train(batchSize, thetaFilename);
  }

  profile.toc();

  return 0;
}

void signalHandler(int param) {
  cout << RED "[Interrupted]" COLOREND << " aborted by user." << endl;
  cout << ORANGE "[Logging]" COLOREND << " saving configuration and experimental results..." << endl;

  //dtwdiag::saveTheta();
  cout << GREEN "[Done]" COLOREND << endl;

  exit(-1);
}


void regSignalHandler () {
  if (signal (SIGINT, signalHandler) == SIG_ERR)
    cerr << "Cannot catch signal" << endl;
}
