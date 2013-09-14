#include <string>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <map>

#include <pbar.h>
#include <utility.h>
#include <array.h>
#include <cmdparser.h>

#include <vulcan-hmm.h>
#include <vulcan-archive.h>

#include <libutility/include/utility.h>
#include <libfeature/include/feature.h>

using namespace std;
using namespace vulcan;

typedef vector<DoubleVector> FeatureSeq;
typedef std::pair<size_t, size_t> Phone; 
map<size_t, string> getPhoneMapping(string filename);
void print(FILE* p, const FeatureSeq& fs);
int load(string alignmentFile, string modelFile, map<string, vector<Phone> >& phoneLabels);
std::string exec(string cmd);

map<size_t, string> pMap;

int main(int argc, char* argv[]) {

  CmdParser cmdParser(argc, argv);
  cmdParser.regOpt("-a", "filename of alignments"); 
  cmdParser.regOpt("-p", "phone table");
  cmdParser.regOpt("-m", "model");

  if(!cmdParser.isOptionLegal())
    cmdParser.showUsageAndExit();

  string alignmentFile = cmdParser.find("-a");
  string phoneTableFile = cmdParser.find("-p");
  string modelFile = cmdParser.find("-m");

  cout << "alignmentFile:  " << alignmentFile << endl;
  cout << "phoneTableFile: " << phoneTableFile << endl;
  cout << "modelFile:      " << modelFile << endl;

  pMap = getPhoneMapping(phoneTableFile);

  map<string, vector<Phone> > phoneLabels;
  int nInstance = load(alignmentFile, modelFile, phoneLabels);

  FILE* fptr = fopen("/media/Data1/LectureDSP_script/feat/train.39.ark", "r");

  VulcanUtterance vUtterance;
  int counter = 0;
  map<size_t, vector<FeatureSeq> > phoneInstances;

  while (vUtterance.LoadKaldi(fptr)) {

    string docId = vUtterance.fid();
    if ( phoneLabels.count(docId) == 0 )
      continue;

    const FeatureSeq& fs = vUtterance._feature;
    const vector<Phone>& phoneLbl = phoneLabels[docId];

    auto offset = fs.begin();
    for (size_t i=0; i<phoneLbl.size(); ++i) {
      size_t phoneIdx = phoneLbl[i].first;
      size_t nFrame = phoneLbl[i].second;

      FeatureSeq fs(nFrame);
      std::copy(offset, offset + nFrame, fs.begin());
      offset += nFrame;
	
      phoneInstances[phoneIdx].push_back(fs);
    }
  }

  size_t nInstanceC = 0;
  for (auto i=phoneInstances.cbegin(); i != phoneInstances.cend(); ++i)
    nInstanceC += i->second.size();
  cout << "nInstanceC = " << nInstanceC << endl;
  assert(nInstanceC == nInstance);
  cout << "Checking total phone instances... " GREEN "[Good]" COLOREND << endl;

  string dir = "data/phone_instances/";
  VulcanUtterance tmpInst;

  nInstanceC = 0;
  for (auto i=phoneInstances.cbegin(); i != phoneInstances.cend(); ++i) {

    string phone = pMap[i->first];
    string folder = "data/mfcc/" + phone;
    const vector<FeatureSeq>& fSeqs = i->second;

    string ret = exec("mkdir -p " + folder);

    /*string filename = dir + phone + ".txt";
    FILE* pFile = fopen(filename.c_str(), "w");*/

    ProgressBar pBar("Saving phone instances for " GREEN + phone + COLOREND "\t...");
    for (size_t i=0; i<fSeqs.size(); ++i) {
      pBar.refresh(double (i+1) / fSeqs.size());
      tmpInst._feature = fSeqs[i];
      tmpInst.SaveHtk(folder + "/" + int2str(i) + ".mfc", false);
      /*fprintf(pFile, "["); print(pFile, fSeqs[i]);
      fprintf(pFile, "]");*/
    }
    nInstanceC += fSeqs.size();

    // fclose(pFile);
  }
  cout << "nInstanceC = " << nInstanceC << endl;

  int nMfccFiles = str2int(exec("ls data/mfcc/* | wc -l"));
  string msg = (nMfccFiles == nInstance) ? GREEN "GOOD" COLOREND : ORANGE "BAD" COLOREND;
  cout << "Checking total phone instances... " << msg << endl;

  cout << "[Done]" << endl;

  return 0;
}

/*int exec(string cmd) {
  system(cmd.c_str());
}*/

void print(FILE* p, const FeatureSeq& fs) {
  for (size_t j=0; j<fs.size(); ++j)
    fs[j].fprintf(p);
}

map<size_t, string> getPhoneMapping(string filename) {

  map<size_t, string> pMap;

  fstream file(filename);

  string line;
  while( std::getline(file, line) ) {
    vector<string> sub = split(line, ' ');
    string p = sub[0];
    size_t idx = str2int(sub[1]);
    pMap[idx] = p;
  }

  return pMap;
}

int load(string alignmentFile, string modelFile, map<string, vector<Phone> >& phoneLabels) {
  
  fstream file(alignmentFile.c_str());

  VulcanHmm vHmm;
  vHmm.LoadKaldiModel(modelFile);

  Array<string> documents;
  vector<size_t> lengths;

  string line;
  while( std::getline(file, line) ) {

    vector<string> substring = split(line, ' ');

    string docId = substring[0];
    //cout << GREEN << "=================== " << docId << " ===================" << COLOREND << endl;

    int prevPhoneId = -1;
    int prevStateId = -1;
    int nFrame = 0;

    for (size_t j=1; j<substring.size(); ++j) {
      size_t transID = str2int(substring[j]);
      size_t phoneId = vHmm.GetPhoneForTransId(transID);
      size_t stateId = vHmm.GetStateForTransId(transID);
      size_t c = vHmm.GetClassForTransId(transID);

      // Find a new phone !! Either because different phoneId or a back-transition of state in a HMM
      if (phoneId != prevPhoneId || (phoneId == prevPhoneId && stateId < prevStateId) ) {

	// TODO Push the previous phone instance into phoneLabels.
	if (prevPhoneId != -1) 
	  phoneLabels[docId].push_back(Phone(prevPhoneId, nFrame));

	nFrame = 1;
      }
      else {
	++nFrame;
      }

      /*if (phoneId != prevPhoneId)
	cout << GREEN << " + " << COLOREND << endl;
      cout << pMap[phoneId] << "(" << c << ") "; */

      prevPhoneId = phoneId;
      prevStateId = stateId;
    }
    //cout << endl;
    
    const vector<Phone>& p = phoneLabels[docId];
    /*cout << BLUE "-------------------------------------------------------" COLOREND << endl;
    for (size_t j=0; j<phoneLabels[docId].size(); ++j) 
      cout << pMap[p[j].first] << "(" << p[j].second << ") ";
    cout << endl;
    cout << "====================================================" << endl << endl;
    */

    documents.push_back(substring[0]);
    lengths.push_back(substring.size() - 1);
  }

  cout << "# of documents = " << documents.size() << endl;

  int nInstance = 0;
  for (auto i=phoneLabels.cbegin(); i != phoneLabels.cend(); ++i)
    nInstance += i->second.size();
  
  cout << "# of total phone instances = " << nInstance << endl;
  return nInstance;
}

std::string exec(string cmd) {
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    return "ERROR";

  char buffer[128];
  std::string result = "";

  try {
    while(!feof(pipe)) {
      if(fgets(buffer, 128, pipe) != NULL)
	result += buffer;
    }
  } catch (...) {
    std::cerr << "[Warning] Exception caught in " << __FUNCTION__ << endl;
  }

  pclose(pipe);
  return result;
}
