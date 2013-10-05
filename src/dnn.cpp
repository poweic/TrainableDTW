#include <dnn.h>

vec loadvector(string filename) {
  Array<float> arr(filename);
  vec v(arr.size());
  foreach (i, arr)
    v[i] = arr[i];
  return v;
}

namespace blas {
  vec rand(size_t size) {
    vec v(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    foreach (i, v)
      v[i] = dis(gen);

    return v;
  }

  vec softmax(const vec& x) {
    vec s(x.size());

    foreach (i, s)
      s[i] = exp(x[i]);

    auto denominator = 1.0 / vecsum(s);
    foreach (i, s)
      s[i] *= denominator;

    return s;
  }

  mat softmax(const mat& m) {
    // TODO
    return m;
  }

  vec sigmoid(const vec& x) {
    vec s(x.size());
    std::transform(x.begin(), x.end(), s.begin(), fn::sigmoid<float>());
    return s;
  }

  vec b_sigmoid(const vec& x) {
    vec s(x.size() + 1);
    std::transform(x.begin(), x.end(), s.begin(), fn::sigmoid<float>());
    s[s.size() - 1] = 1.0;
    return s;
  }
};


vec DNN::feedForward(const vec& x) {

  // Init with one extra element, which is bias
  _v_output[0].resize(x.size() + 1);
  std::copy (x.begin(), x.end(), _v_output[0].begin());

  for (size_t i=1; i<_v_output.size(); ++i)
    _v_output[i] = blas::b_sigmoid(_v_output[i-1] * _weights[i-1]);

  return _v_output.back();
}

mat DNN::feedForward(const mat& batch_x) {
  // TODO
  return mat();
}

vec DNN::backPropagate(const vec& x) {
  // TODO
  return vec();
}

mat DNN::backPropagate(const mat& batch_p) {
  // TODO
  return mat();
}

vec operator*(const vec& x, const vec& y) {
  vec z(x.size());
  std::transform (x.begin(), x.end(), y.begin(), z.begin(), std::multiplies<float>());
  return z;
}

vec operator*(const mat& A, const vec& col_vector) {

  vec y(A.getRows());
  int cols = A.getCols();

  foreach (i, y) {
    for (size_t j=0; j<cols; ++j)
      y[i] += col_vector[j] * A[i][j];
  }

  return y;
}

vec operator*(const vec& row_vector, const mat& A) {

  vec y(A.getCols());
  int rows = A.getRows();

  foreach (i, y) {
    for (size_t j=0; j<rows; ++j)
      y[i] += row_vector[j] * A[j][i];
  }

  return y;
}
