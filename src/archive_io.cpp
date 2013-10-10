#include <archive_io.h>

// **************************************
// ***** Load Kaldi Feature Archive *****
// **************************************
size_t loadFeatureArchive(const string& featArk, const map<string, vector<Phone> >& phoneLabels, map<size_t, vector<FeatureSeq> >& phoneInstances) {

  FILE* fptr = fopen(featArk.c_str(), "r");
  VulcanUtterance vUtterance;
  int counter = 0;
  while (vUtterance.LoadKaldi(fptr)) {

    string docId = vUtterance.fid();
    if ( phoneLabels.count(docId) == 0 )
      continue;

    const FeatureSeq& fs = vUtterance._feature;
    const vector<Phone>& phoneLbl = phoneLabels.find(docId)->second;

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

  size_t nInstance = 0;
  for (auto i=phoneInstances.cbegin(); i != phoneInstances.cend(); ++i)
    nInstance += i->second.size();
  return nInstance;
}

// ***************************************
// ***** Save Features as MFCC files *****
// ***************************************
size_t saveFeatureAsMFCC(const map<size_t, vector<FeatureSeq> >& phoneInstances, const vector<string> &phones, const string& dir) {
  VulcanUtterance tmpInst;

  size_t nInstanceC = 0;
  for (auto i=phoneInstances.cbegin(); i != phoneInstances.cend(); ++i) {

    string phone = phones[i->first];
    string folder = dir + phone;
    const vector<FeatureSeq>& fSeqs = i->second;

    string ret = exec("mkdir -p " + folder);

    ProgressBar pBar("Saving phone instances for " GREEN + phone + COLOREND "\t...");
    for (size_t i=0; i<fSeqs.size(); ++i) {
      pBar.refresh(double (i+1) / fSeqs.size());
      tmpInst._feature = fSeqs[i];
      tmpInst.SaveHtk(folder + "/" + int2str(i) + ".mfc", false);
    }
    nInstanceC += fSeqs.size();
  }
  cout << "nInstanceC = " << nInstanceC << endl;

  int nMfccFiles = str2int(exec("ls " + dir + "* | wc -l"));
  return nMfccFiles;
}

// *********************************
// ***** Load Phone Alignments *****
// *********************************
int load(string alignmentFile, string modelFile, map<string, vector<Phone> >& phoneLabels, bool dump) {
  
  fstream file(alignmentFile.c_str());

  VulcanHmm vHmm;
  vHmm.LoadKaldiModel(modelFile);

  Array<string> documents;
  vector<size_t> lengths;

  string phoneAlignment;

  string line;
  while( std::getline(file, line) ) {

    vector<string> substring = split(line, ' ');

    string docId = substring[0];
    phoneAlignment += docId + " ";

    size_t prevPhoneId = -1;
    size_t prevStateId = -1;
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

      phoneAlignment += int2str(phoneId) + " ";

      prevPhoneId = phoneId;
      prevStateId = stateId;
    }
    phoneAlignment += "\n";
    
    const vector<Phone>& p = phoneLabels[docId];

    documents.push_back(substring[0]);
    lengths.push_back(substring.size() - 1);
  }

  if (dump)
    cout << phoneAlignment << endl;

  int nInstance = 0;
  for (auto i=phoneLabels.cbegin(); i != phoneLabels.cend(); ++i)
    nInstance += i->second.size();
  
  return nInstance;
}

// *****************************************************
// ***** Load Phone Mappings from File into vector *****
// *****************************************************
vector<string> getPhoneMapping(string filename) {

  vector<string> phones;

  fstream file(filename);

  string line;
  while( std::getline(file, line) ) {
    vector<string> sub = split(line, ' ');
    string p = sub[0];
    //size_t idx = str2int(sub[1]);
    phones.push_back(p);
  }

  return phones;
}

// **************************************
// ***** Print Out Feature Sequence *****
// **************************************
void print(FILE* p, const FeatureSeq& fs) {
  for (size_t j=0; j<fs.size(); ++j)
    fs[j].fprintf(p);
}

