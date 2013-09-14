#include <cdtw.h>

namespace DtwUtil {
  void CumulativeDtwRunner::CalScoreTable() {
    // q == 0
    for (int d = 0; d < dL_; ++d)
      score_(0, d) = pdist(0, d);

    // d == 0
    for (int q = 1; q < qL_; ++q)
      score_(q, 0) = score_(q - 1, 0) + pdist(q, 0);

    // interior points
    for (int d = 1; d < dL_; ++d) {
      for (int q = 1; q < qL_; ++q) {

	// TODO Use softmax to do the Cumulative DTW
	double s1 = score_(q, d-1) + pdist(q, d),
	       s2 = score_(q-1, d) + pdist(q, d),
	       s3 = score_(q-1, d-1) + pdist(q, d);

	double s = softmin(s1, s2, s3);
	score_(q, d) = s;
	//cout << "(" << s1 << ", " << s2 << ", " << s3 << ") => s = " << s << "\n";
      }
    }
    double score = score_(qL_ - 1, dL_ - 1);
    double normalizer = qL_ + dL_;
    cout << "qL_        : \33[32m" << qL_ << "\33[0m" << endl;
    cout << "dL_        : \33[32m" << dL_ << "\33[0m" << endl;
    cout << "Score      : \33[32m" << score << "\33[0m" << endl;
    cout << "Normalizer : \33[32m" << normalizer << "\33[0m" << endl;
    cout << "Normalized : \33[32m" << score / normalizer << "\33[0m" << endl;

  }
};
