#ifndef __DEVICE_MATRIX_H__
#define __DEVICE_MATRIX_H__

#include <matrix.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
using namespace std;

#define CCE(x) checkCudaErrors(x)

#define host_matrix Matrix2D
#define STRIDE (sizeof(T) / sizeof(float))

class CUBLAS_HANDLE {
public:
  CUBLAS_HANDLE()  { CCE(cublasCreate(&_handle)); }
  ~CUBLAS_HANDLE() { CCE(cublasDestroy(_handle)); }

  cublasHandle_t& get() { return _handle; }
private:
  cublasHandle_t _handle;
};

template <typename T>
class device_matrix {
public:
  device_matrix(size_t r, size_t c): _rows(r), _cols(c), _data(NULL) {
    _init();
  }

  // Copy Constructor 
  device_matrix(const device_matrix<T>& source): _rows(source._rows), _cols(source._cols), _data(NULL) {
    //cout << "In Copy Constructor" << endl;
    _init();

    cublasScopy(device_matrix<T>::_handle.get(), _rows * _cols, source._data, STRIDE, _data, STRIDE);
  }

  // Constructor from Host Matrix
  device_matrix(const host_matrix<T>& h_matrix): _rows(h_matrix.getRows()), _cols(h_matrix.getCols()), _data(NULL) {

    // Convert T** to column major using transpose
    host_matrix<T> cm_h_matrix(~h_matrix);
    _init();

    size_t n = _rows * _cols;

    T* h_data = new T[n];
    for (size_t i=0; i<_cols; ++i)
      memcpy(h_data + i*_rows, cm_h_matrix[i], sizeof(T) * _rows);

    CCE(cublasSetVector(n, sizeof(T), h_data, STRIDE, _data, STRIDE));

    delete h_data;
  }

  // Math Operation on Matrix 
  device_matrix<T> operator * (const device_matrix<T>& rhs) {

    size_t m = this->_rows;
    size_t n = rhs._cols;
    device_matrix<T> result(m, n);

    size_t k = this->_cols;

    float alpha = 1.0;
    float beta = 0;

    int lda = this->_rows;
    int ldb = rhs._rows;
    int ldc = result._rows;

    cublasStatus_t status;
    status = cublasSgemm(device_matrix<T>::_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, this->_data, lda, rhs._data, ldb, &beta, result._data, ldc);

    CCE(status);

    return result;
  }

  ~device_matrix() {
    CCE(cudaFree(_data));
  }

  // ===========================
  // ===== Other Functions =====
  // ===========================

  operator host_matrix<T>() {

    host_matrix<T> cm_h_matrix(_cols, _rows);

    size_t n = _rows * _cols;
    T* h_data = new T[n];

    CCE(cublasGetVector(n, sizeof(T), _data, STRIDE, h_data, STRIDE));

    for (size_t i=0; i<_cols; ++i)
      memcpy(cm_h_matrix[i], h_data + i*_rows, sizeof(T) * _rows);

    delete h_data;

    return ~cm_h_matrix;
  }


  template <typename S>
  friend void swap(device_matrix<S>& lhs, device_matrix<S>& rhs);

  // Operator Assignment:
  // call copy constructor first, and swap with the temp variable
  device_matrix<T>& operator = (host_matrix<T> rhs) {
    swap(*this, rhs);
    return *this;
  }

  void _init() {
    CCE(cudaMalloc((void **)&_data, _rows * _cols * sizeof(T)));
  }

  /*void print() {
    for (size_t i=0; i<_rows; ++i) {
      for (size_t j=0; j<_cols; ++j)
	printf("%.1f ", _data[i * _cols + j]);
      printf("\n");
    }
  }*/

  CUBLAS_HANDLE& getCublasHandle() { return _handle; }

  size_t getRows() const { return _rows; }
  size_t getCols() const { return _cols; }

  static CUBLAS_HANDLE _handle;

private:

  size_t _rows;
  size_t _cols;
  T* _data;
};

template <typename T>
void swap(device_matrix<T>& lhs, device_matrix<T>& rhs) {
  using std::swap;
  swap(lhs._rows, rhs._rows);
  swap(lhs._cols, rhs._cols);
  swap(lhs._data, rhs._data);
}

// In a class template, when performing implicit instantiation, the 
// members are instantiated on demand. Since the code does not use the
// static member, it's not even instantiated in the whole application.
template <typename T>
CUBLAS_HANDLE device_matrix<T>::_handle;

template <typename T>
void sgemm(device_matrix<T>& A, device_matrix<T>& B, device_matrix<T>& C) {
  // TODO
  // Perform C = αA*B + βC, not transpose on A and B

}


#endif // __DEVICE_MATRIX_H__
