#include <archive_io.h>

// **************************************
// ***** Load Kaldi Feature Archive *****
// **************************************
size_t loadFeatureArchive(const string& featArk, const map<string, vector<Phone> >& phoneLabels, map<size_t, vector<FeatureSeq> >& phoneInstances) {

  FILE* fptr = fopen(featArk.c_str(), "r");
  VulcanUtterance vUtterance;
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

void loadFeatureArchive(string filename, float* &data, unsigned int* &offset, int& N, int& dim) {

  vector<FeatureSeq> featureSeqs;

  FILE* fptr = fopen(filename.c_str(), "r");
  vulcan::VulcanUtterance vUtterance;
  while (vUtterance.LoadKaldi(fptr))
    featureSeqs.push_back(vUtterance._feature);
  fclose(fptr);

  N = featureSeqs.size();
  dim = featureSeqs[0][0].size();

  offset = new unsigned int[N + 1];
  offset[0] = 0;
  for (int i=1; i<N+1; ++i) {
    size_t prevLength = featureSeqs[i-1].size();
    offset[i] = offset[i-1] + prevLength * dim;
  }

  size_t totalLength = offset[N];
  data = new float[totalLength];

  for (int i=0; i<N; ++i) {
    unsigned int begin = offset[i];
    unsigned int  end  = offset[i+1];
    float* d = &data[begin];
    int length = (end - begin) / dim;

    for (int j=0; j<length; ++j) 
      for (int k=0; k<dim; ++k)
	d[j * dim + k] = featureSeqs[i][j]._data->data[k]; 
  }
}

// ***************************************
// ***** Save Features as MFCC files *****
// ***************************************

void save(const FeatureSeq& featureSeq, const string& filename) {
  VulcanUtterance tmpInst;
  tmpInst._feature = featureSeq;
  tmpInst.SaveHtk(filename, false);
}

size_t saveFeatureAsMFCC(const map<size_t, vector<FeatureSeq> >& phoneInstances, const vector<string> &phones, string dir) {
  dir += "/";

  size_t nInstanceC = 0;
  for (auto i=phoneInstances.cbegin(); i != phoneInstances.cend(); ++i) {

    string phone = phones[i->first];
    string folder = dir + phone;
    const vector<FeatureSeq>& fSeqs = i->second;

    string ret = exec("mkdir -p " + folder);

    ProgressBar pBar("Saving phone instances for \t" GREEN + phone + COLOREND "\t...");
    for (size_t i=0; i<fSeqs.size(); ++i) {
      pBar.refresh(double (i+1) / fSeqs.size());
      save(fSeqs[i], folder + "/" + int2str(i) + ".mfc");
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

    int prevPhoneId = -1;
    int prevStateId = -1;
    int nFrame = 0;

    for (size_t j=1; j<substring.size(); ++j) {
      int transID = str2int(substring[j]);
      int phoneId = vHmm.GetPhoneForTransId(transID);
      int stateId = vHmm.GetStateForTransId(transID);
      //size_t c = vHmm.GetClassForTransId(transID);

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
    
    //const vector<Phone>& p = phoneLabels[docId];

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

