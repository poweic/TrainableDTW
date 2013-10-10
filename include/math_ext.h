#include <vector>
#include <time.h>
#include <cstdlib>

#include <matrix.h>
#include <functional.inl>

namespace ext {
  // ==================================
  // ===== First Order Difference =====
  // ==================================
  template <typename T>
  vector<T> diff1st(const vector<T>& v) {
    vector<T> diff(v.size() - 1);
    foreach (i, diff)
      diff[i] = v[i+1] - v[i];
    return diff;
  }
  
  // =========================
  // ===== Random Vector =====
  // =========================
  
  inline double rand01() {
    srand(time(NULL));
    return (double) rand() / (double) RAND_MAX;
  }

  template <typename T>
  vector<T> rand(size_t size) {
    vector<T> v(size);

    foreach (i, v)
      v[i] = rand01();

    return v;
  }

  template <typename T>
  void rand(Matrix2D<T>& m) {

    for (size_t i=0; i<m.getRows(); ++i)
      for (size_t j=0; j<m.getCols(); ++j)
	m[i][j] = rand01();
  }

  // ===================
  // ===== SoftMax =====
  // ===================
  template <typename T>
  vector<T> softmax(const vector<T>& x) {
    vector<T> s(x.size());

    foreach (i, s)
      s[i] = exp(x[i]);

    T denominator = 1.0 / vecsum(s);
    foreach (i, s)
      s[i] *= denominator;

    return s;
  }

  // ============================
  // ===== Sigmoid Function =====
  // ============================
  template <typename T>
  vector<T> sigmoid(const vector<T>& x) {
    vector<T> s(x.size());
    std::transform(x.begin(), x.end(), s.begin(), func::sigmoid<float>());
    return s;
  }

  // ================================
  // ===== Biased after Sigmoid =====
  // ================================
  template <typename T>
  vector<T> b_sigmoid(const vector<T>& x) {
    vector<T> s(x.size() + 1);
    std::transform(x.begin(), x.end(), s.begin(), func::sigmoid<float>());
    s[s.size() - 1] = 1.0;
    return s;
  }
  
};
