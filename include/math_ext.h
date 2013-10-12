#ifndef __MATH_EXT_H_
#define __MATH_EXT_H_

#include <vector>
#include <time.h>
#include <cstdlib>

#include <matrix.h>
#include <functional.inl>

namespace ext {
  // ========================
  // ===== Save as File =====
  // ========================
  template <typename T>
  void save(const vector<T>& v, string filename) {
    ofstream fs(filename.c_str());

    foreach (i, v)
      fs << v[i] << endl;

    fs.close();
  }

  // ==========================
  // ===== Load from File =====
  // ==========================
  template <typename T>
  void load(vector<T>& v, string filename) {
    v.clear();

    ifstream fs(filename.c_str());

    T t;
    while (fs >> t) 
      v.push_back(t);

    fs.close();
  }

  // =================================
  // ===== Summation over Vector =====
  // =================================
  template <typename T>
  T sum(const vector<T>& v) {
    T s = 0;
    foreach (i, v)
      s += v[i];
    return s;
  }

  // =================================
  // ===== Summation over Vector =====
  // =================================
  template <typename T>
  T sum(const Matrix2D<T>& m) {
    T s = 0;
    range ( i, m.getRows() )
      range ( j, m.getCols() )
	s += m[i][j];
      
    return s;
  }

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
  T rand01() {
    return (T) ::rand() / (T) RAND_MAX;
  }
  
  template <typename T>
  vector<T> rand(size_t size) {
    srand(time(NULL));
    vector<T> v(size);

    foreach (i, v)
      v[i] = rand01<T>();

    return v;
  }

  template <typename T>
  void rand(Matrix2D<T>& m) {
    srand(time(NULL));

    for (size_t i=0; i<m.getRows(); ++i)
      for (size_t j=0; j<m.getCols(); ++j)
	m[i][j] = rand01<T>();
  }

  // ===================
  // ===== SoftMax =====
  // ===================
  template <typename T>
  vector<T> softmax(const vector<T>& x) {
    vector<T> s(x.size());

    foreach (i, s)
      s[i] = exp(x[i]);

    T denominator = 1.0 / ext::sum(s);
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
    std::transform(x.begin(), x.end(), s.begin(), func::sigmoid<T>());
    return s;
  }

  // ================================
  // ===== Biased after Sigmoid =====
  // ================================
  template <typename T>
  vector<T> b_sigmoid(const vector<T>& x) {
    vector<T> s(x.size() + 1);
    std::transform(x.begin(), x.end(), s.begin(), func::sigmoid<T>());
    s.back() = 1.0;
    return s;
  }
  
};

#endif
