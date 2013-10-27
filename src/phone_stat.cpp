#include <phone_stat.h>

void deduceCompetitivePhones(const Array<string>& phones, const Matrix2D<double>& scores) {
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

double objective(const Matrix2D<double>& scores) {
  double obj = 0;
  for (size_t i=0; i<scores.getRows(); ++i) {
    obj += scores[i][i];
    for (size_t j=0; j<i; ++j)
      obj -= scores[i][j];
  }
  return obj;
}

Matrix2D<double> comparePhoneDistances(const Array<string>& phones, const string& score_dir) {

  Matrix2D<double> scores(phones.size(), phones.size());

  foreach (i, phones) {
    if (i <= 1) continue;
    string phone1 = phones[i];

    foreach (j, phones) {
      if (j <= 1) continue;
      if (j > i) break;
      string phone2 = phones[j];

      string file = score_dir + int2str(i) + "-" + int2str(j) + ".Matrix2D<double>";
      double avg = average(Matrix2D<double>(file));

      //printf("avg = %.4f between phone #%d : %6s and phone #%d : %6s", avg, i, phone1.c_str(), j, phone2.c_str());
      scores[i][j] = scores[j][i] = avg;
    }
  }

  return scores;
}

Matrix2D<double> statistics(const Array<string>& phones) {
  vector<Array<string> > lists(phones.size());

  // Run statistics of nPairs to get the distribution
  double nPairsInTotal = 0;
  Matrix2D<double> nPairs(phones.size(), phones.size());

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

void computeBetweenPhoneDistance(const Array<string>& phones, const string& MFCC_DIR, size_t N, const string& score_dir) {
  vector<Array<string> > lists(phones.size());

  // const size_t MAX_STATIONARY_ITR = 1000;
  // size_t nItrStationary = 0;

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

      Matrix2D<double> score(rows, cols);
    
      foreach (m, lists[i]) {
	if (m >= N) break;
	foreach (n, lists[j]) {
	  if (n >= N) break;

	  string f1 = MFCC_DIR + phone1 + "/" + lists[i][m];
	  string f2 = MFCC_DIR + phone2 + "/" + lists[j][n];
	  //score[m][n] = dtwdiag::dtw(f1, f2);
	}
      }

      string file = score_dir + int2str(i) + "-" + int2str(j) + ".Matrix2D<double>";
      score.saveas(file);
    }
  }

}

double average(const Matrix2D<double>& m, double MIN) {
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

void evaluate(bool reevaluate, const Array<string>& phones, string MFCC_DIR, size_t N, string matFile, string scoreDir) {

  scoreDir += "/";
  exec("mkdir -p " + scoreDir);

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

