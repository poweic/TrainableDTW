#ifndef __VECTOR_BLAS_H__
#define __VECTOR_BLAS_H__

#include <utility.h>
#include <vector>
#include <matrix.h>
#include <functional>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#define ASSERT_NOT_SCALAR(T) {static_assert(std::is_scalar<T>::value, "val must be scalar");} 
#else
#define ASSERT_NOT_SCALAR(T) {}
#endif

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

using namespace std;

// ===============================
// ===== Template Functional =====
// ===============================
namespace func {

  template <typename T>
  struct apx {
    const T a;
    apx(T _a) : a(_a) {}
    __host__ __device__ T operator() (const T& x) const { return a + x; }
  };

  template <typename T>
  struct amx {
    const T a;
    amx(T _a) : a(_a) {}
    __host__ __device__ T operator() (const T& x) const { return a - x; }
  };

  template <typename T>
  struct ax {
    const T a;
    ax(T _a) : a(_a) {}
    __host__ __device__ T operator() (const T& x) const { return a * x; }
  };

  template <typename T>
  struct adx {
    const T a;
    adx(T _a) : a(_a) {}
    __host__ __device__ T operator() (const T& x) const { return a / x; }
  };

  template <typename T>
  struct square {
    __host__ __device__ T operator() (const T& x) const { return x * x; }
  };
};

// =====================================
// ===== Matrix - Vector Operators =====
// =====================================
template <typename T>
Matrix2D<T> operator * (const vector<T>& col_vector, const vector<T>& row_vector) {

  Matrix2D<T> m(col_vector.size(), row_vector.size());

  foreach (i, col_vector)
    foreach (j, row_vector)
      m[i][j] = col_vector[i] * row_vector[j];

  return m;
}

template <typename T>
vector<T> operator & (const vector<T>& x, const vector<T>& y) {
  assert(x.size() == y.size());
  vector<T> z(x.size());
  std::transform (x.begin(), x.end(), y.begin(), z.begin(), std::multiplies<float>());
  return z;
}

template <typename T>
vector<T> operator * (const Matrix2D<T>& A, const vector<T>& col_vector) {
  assert(A.getCols() == col_vector.size());

  vector<T> y(A.getRows());
  int cols = A.getCols();

  foreach (i, y) {
    for (size_t j=0; j<cols; ++j)
      y[i] += col_vector[j] * A[i][j];
  }

  return y;
}

template <typename T>
vector<T> operator * (const vector<T>& row_vector, const Matrix2D<T>& A) {
  assert(row_vector.size() == A.getRows());

  vector<T> y(A.getCols());
  int rows = A.getRows();

  foreach (i, y) {
    for (size_t j=0; j<rows; ++j)
      y[i] += row_vector[j] * A[j][i];
  }

  return y;
}

#define VECTOR std::vector
#define WHERE std
#include <blas.inl>
#undef VECTOR
#undef WHERE

void blas_testing_examples();

#endif // __VECTOR_BLAS_H__
