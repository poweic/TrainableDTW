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

//void initModel(Model& model, size_t feat_dim, size_t nLayer, size_t nHiddenNodes, float lr);
Array<string> getPhoneList(string filename);
void signalHandler(int param);
void regSignalHandler();

int main (int argc, char* argv[]) {
  
  CmdParser cmdParser(argc, argv);
  cmdParser
    .addGroup("Generic options")
    .add("-p", "Choose either \"selftest\", \"train\".")
    .add("--phone-mapping", "The mapping of phones", false, "data/phones.txt");

  cmdParser
    .addGroup("Distance measure options")
    .add("--eta", "Specify the coefficient in the smoothing minimum", false, "-2")
    .add("--weight", "Specify the weight between intra-phone & inter-phone", false, "0.065382482");

  cmdParser
    .addGroup("Training options")
    .add("--model", "choose a distance model, either \"dnn\" or \"diag\"", false, "dnn")
    .add("--batch-size", "number of training samples per batch", false, "1000")
    .add("--learning-rate", "learning rate", false, "0.0001")
    .add("--theta-output", "choose a file to save theta", false, ".theta.restore");

  cmdParser
    .addGroup("Training Corpus options:")
    .add("--feat-dim", "dimension of feature vector (ex: 39 for mfcc)", false, "39")
    .add("--feat-dir", "root directory of feature files ex: data/mfcc/", false, "/share/mlp_posterior/gaussian_posterior_noprior_no_log/");

  cmdParser
    .addGroup("Deep Neural Network options:")
    .add("--layers", "Number of hidden layer in both PP and DTW", false, "3")
    .add("--hidden-nodes", "Number of hidden nodes per hidden layer", false, "64");
  
  cmdParser
    .addGroup("Evaluation options")
    .add("-d", "directory for saving/loading scores", false)
    .add("-o", "filename for scores matrix", false);

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  // Parsering Command Arguments
  string phase = cmdParser.find("-p");

  size_t batchSize	  = str2int(cmdParser.find("--batch-size"));
  string feat_dir   	  = cmdParser.find("--feat-dir");

  string phones_filename  = cmdParser.find("--phone-mapping");
  Array<string> phones	  = getPhoneList(phones_filename);

  string matFile	  = cmdParser.find("-o");
  double eta		  = str2double(cmdParser.find("--eta"));
  float  lr		  = str2float(cmdParser.find("--learning-rate"));
  size_t nHiddenLayer	  = str2int(cmdParser.find("--layers"));
  size_t nHiddenNodes	  = str2int(cmdParser.find("--hidden-nodes"));
  string m		  = cmdParser.find("--model");
  string thetaFilename	  = cmdParser.find("--theta-output");

  size_t feat_dim	  = str2int(cmdParser.find("--feat-dim"));
  float intra_inter_weight= str2double(cmdParser.find("--weight"));

  Profile profile;
  profile.tic();

  SMIN::eta = eta;
  Corpus corpus("data/phones.txt", feat_dir);

  if (m == "dnn") {
    dtwdnn dnn(feat_dim, intra_inter_weight, lr, nHiddenLayer, nHiddenNodes);

    if (phase == "selftest")
      dnn.selftest(corpus);
    else if (phase == "train")
      dnn.train(corpus, batchSize);
  }
  else if (m == "diag") {

    dtwdiag diag(feat_dim, intra_inter_weight, lr, thetaFilename);

    if (phase == "selftest")
      diag.selftest(corpus);
    else if (phase == "train")
      diag.train(corpus, batchSize);
  }

  profile.toc();

  return 0;
}

void signalHandler(int param) {
  cout << RED "[Interrupted]" COLOREND << " aborted by user." << endl;
  cout << ORANGE "[Logging]" COLOREND << " saving configuration and experimental results..." << endl;

  cout << GREEN "[Done]" COLOREND << endl;
  exit(-1);
}


void regSignalHandler () {
  if (signal (SIGINT, signalHandler) == SIG_ERR)
    cerr << "Cannot catch signal" << endl;
}
