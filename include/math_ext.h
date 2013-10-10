#include <vector>
#include <matrix.h>
#include <random>

namespace blas {
  namespace fn {
    template <typename T>
    struct sigmoid {
      T operator()(T x) { return 1 / ( 1 + exp(-x) ); }
    };
  };
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
  template <typename T>
  vector<T> rand(size_t size) {
    vector<T> v(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);

    foreach (i, v)
      v[i] = dist(gen);

    return v;
  }

  template <typename T>
  Matrix2D<T> rand(size_t m, size_t n) {

    Matrix2D<T> R(m, n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);

    for (size_t i=0; i<m; ++i)
      for (size_t j=0; j<n; ++j)
	R[i][j] = dist(gen);

    return R;
  }

  // ===================
  // ===== SoftMax =====
  // ===================
  template <typename T>
  vector<T> softmax(const vector<T>& x) {
    vector<T> s(x.size());

    foreach (i, s)
      s[i] = exp(x[i]);

    auto denominator = 1.0 / vecsum(s);
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
    std::transform(x.begin(), x.end(), s.begin(), fn::sigmoid<float>());
    return s;
  }

  // ================================
  // ===== Biased after Sigmoid =====
  // ================================
  template <typename T>
  vector<T> b_sigmoid(const vector<T>& x) {
    vector<T> s(x.size() + 1);
    std::transform(x.begin(), x.end(), s.begin(), fn::sigmoid<float>());
    s[s.size() - 1] = 1.0;
    return s;
  }
  
};
