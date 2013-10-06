#ifndef __VECTOR_BLAS_H__
#define __VECTOR_BLAS_H__

#include <utility.h>
#include <vector>
#include <matrix.h>
#include <random>
#include <functional>
using namespace std;

// =====================================
// ===== Matrix / Vector Operators =====
// =====================================
template <typename T>
Matrix2D<T> operator*(const vector<T>& col_vector, const vector<T>& row_vector) {

  Matrix2D<T> m(row_vector.size(), col_vector.size());

  foreach (i, row_vector)
    foreach (j, col_vector)
      m[i][j] = row_vector[i] * col_vector[j];

  return m;
}

template <typename T>
vector<T> operator&(const vector<T>& x, const vector<T>& y) {
  vector<T> z(x.size());
  std::transform (x.begin(), x.end(), y.begin(), z.begin(), std::multiplies<float>());
  return z;
}

template <typename T>
vector<T> operator*(const Matrix2D<T>& A, const vector<T>& col_vector) {

  vector<T> y(A.getRows());
  int cols = A.getCols();

  foreach (i, y) {
    for (size_t j=0; j<cols; ++j)
      y[i] += col_vector[j] * A[i][j];
  }

  return y;
}

template <typename T>
vector<T> operator*(const vector<T>& row_vector, const Matrix2D<T>& A) {

  vector<T> y(A.getCols());
  int rows = A.getRows();

  foreach (i, y) {
    for (size_t j=0; j<rows; ++j)
      y[i] += row_vector[j] * A[j][i];
  }

  return y;
}

namespace blas {
  namespace fn {
    template <typename T>
    struct sigmoid {
      T operator()(T x) { return 1 / ( 1 + exp(-x) ); }
    };
  };
  // =========================
  // ===== Random Vector =====
  // =========================
  template <typename T>
  vector<T> rand(size_t size) {
    vector<T> v(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    foreach (i, v)
      v[i] = dis(gen);

    return v;
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

// ===========================
// ===== vector * scalar =====
// ===========================
// [1 2 3] * 10 ==> [10 20 30]
template <typename T, typename U>
vector<T> operator * (const vector<T> &v, U val) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");

  T value = (T) val;
  vector<T> v2(v.size());
  foreach (i, v)
    v2[i] = v[i] * value;
  return v2;
}

// ===========================
// ===== scalar * vector =====
// ===========================
// 10 * [1 2 3] ==> [10 20 30]
template <typename T, typename U>
vector<T> operator * (U val, const vector<T> &v) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");
  return v*val;
}
// ===========================
// ===== vector / scalar =====
// ===========================
// [10 20 30] / 10 ==> [1 2 3]
template <typename T, typename U>
vector<T> operator / (const vector<T> &v, U val) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");
  
  T value = (T) val;
  vector<T> v2(v.size());
  foreach (i, v)
    v2[i] = v[i] / value;
  return v2;
}

// =================================
// ======= scalar ./ vector ========
// =================================
// 10 / [1 2 5] ==> [10/1 10/2 10/5]
template <typename T, typename U>
vector<T> operator / (U val, const vector<T> &v) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");
  
  T value = (T) val;
  vector<T> v2(v.size());
  foreach (i, v)
    v2[i] = value / v[i];
  return v2;
}

// ===========================
// ===== scalar + vector =====
// ===========================
// [1 2 3 4] + 5 ==> [6 7 8 9]
template <typename T, typename U>
vector<T> operator + (U val, const vector<T> &v1) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");

  T value = (T) val;
  vector<T> v(v1.size());
  transform(v1.begin(), v1.end(), v.begin(), std::bind1st(std::plus<T>(), value));
  return v;
}

// ===========================
// ===== vector + scalar =====
// ===========================
// [1 2 3 4] + 5 ==> [6 7 8 9]
template <typename T, typename U>
vector<T> operator + (const vector<T> &v1, U val) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");

  T value = (T) val;
  vector<T> v(v1.size());
  transform(v1.begin(), v1.end(), v.begin(), std::bind2nd(std::plus<T>(), value));
  return v;
}

// =============================
// ====== vector + vector ======
// =============================
// [1 2 3] + [2 3 4] ==> [3 5 7]
template <typename T>
vector<T> operator + (const vector<T> &v1, const vector<T> &v2) {
  vector<T> sum(v1.size());
  std::transform(v1.begin(), v1.end(), v2.begin(), sum.begin(), std::plus<T>());
  return sum;
}

// ===========================
// ===== scalar - vector =====
// ===========================
// 5 - [1 2 3 4] ==> [4 3 2 1]
template <typename T, typename U>
vector<T> operator - (U val, const vector<T> &v1) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");

  T value = (T) val;
  vector<T> v(v1.size());
  transform(v1.begin(), v1.end(), v.begin(), std::bind1st(std::minus<T>(), value)); 
  return v;
}

// ===========================
// ===== vector - scalar =====
// ===========================
// [1 2 3 4] - 1 ==> [0 1 2 3]
template <typename T, typename U>
vector<T> operator - (const vector<T> &v1, U val) {
  static_assert(std::is_scalar<U>::value, "val must be scalar");

  T value = (T) val;
  vector<T> v(v1.size());
  transform(v1.begin(), v1.end(), v.begin(), std::bind2nd(std::minus<T>(), value));
  return v;
}

// =============================
// ====== vector - vector ======
// =============================
// [2 3 4] - [1 2 3] ==> [1 1 1]
template <typename T>
vector<T> operator - (const vector<T> &v1, const vector<T> &v2) {
  vector<T> diff(v1.size());
  std::transform(v1.begin(), v1.end(), v2.begin(), diff.begin(), std::minus<T>());
  return diff;
}

/*
void blas_example() {

  vec x(4);
  foreach (i, x)
    x[i] = i+1;

  print(x);

  cout << endl << "1 - [1 2 3 4]" << endl;
  print(1 - x);

  cout << endl << "[1 2 3 4] - 1" << endl;
  print(x - 1);

  cout << endl << "1 + [1 2 3 4]" << endl;
  print(1 + x);

  cout << endl << "[1 2 3 4] + 1" << endl;
  print(x + 1);


  cout << endl << "3 * [1 2 3 4]" << endl;
  print(3 * x);

  cout << endl << "[1 2 3 4] * 4" << endl;
  print(x * 4);

  cout << endl << "12 / [1 2 3 4]" << endl;
  print(12 / x);

  cout << endl << "[1 2 3 4] / 0.5" << endl;
  print(x / 0.5);
} */

#endif // __VECTOR_BLAS_H__
