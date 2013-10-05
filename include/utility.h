#ifndef _UTILITY_H_
#define _UTILITY_H_
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <sys/stat.h>

#include <color.h>
using namespace std;

#define foreach(i, arr) for (size_t i=0; i<arr.size(); ++i)
#define FLOAT_MIN (std::numeric_limits<float>::lowest())

#ifdef DEBUG
  #define debug(token) {cout << #token " = " << token << endl;}
#else
  #define debug(token) {}
#endif

#define mylog(token) {cout << #token " = " << token << endl;}

#define fillwith(arr, val) {std::fill(arr.begin(), arr.end(), val);}
#define fillzero(arr) fillwith(arr, 0)
#define checkNAN(x) assert((x) == (x))

#define __DIVIDER__ "=========================================================="

string int2str(int n);
int str2int(const string& str);
float str2float(const string& str);
double str2double(const string& str);
string getValueStr(string& str);
string join(const vector<string>& arr);

bool isInt(string str);

vector<string> split(const string &s, char delim);
vector<string>& split(const string &s, char delim, vector<string>& elems);

//void pause();

inline bool exists (const string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

template <typename T, typename S>
vector<T> vmax(S a, const vector<T>& v) {
  vector<T> m(v.size());
  foreach (i, m)
    m[i] = MAX(a, v[i]);
  return m;
}

template <typename T, typename S>
vector<T> vmix(S a, const vector<T>& v) {
  vector<T> m(v.size());
  foreach (i, m)
    m[i] = MIN(a, v[i]);
  return m;
}


template <typename T>
T norm(vector<T> v) {
  T sum = 0;
  foreach (i, v) sum += pow(v[i], (T) 2);
  return sqrt(sum);
}

template <typename T>
T vecsum(vector<T> v) {
  T sum = 0;
  foreach (i, v) sum += v[i];
  return sum;
}

template <typename T>
void normalize(vector<T>& v, int type = 2) {
  T n = (type == 2) ? norm(v) : vecsum(v);
  if (n == 0) return;

  T normalizer = 1/n;
  foreach (i, v)
    v[i] *= normalizer;
}

template <typename T>
void print(const vector<T>& v) {
  printf("[");
  foreach (i, v)
    printf("%.2f ", v[i]);
  printf("]\n");
}

template <typename T>
vector<T> operator / (const vector<T> &v, const T c) {
  vector<T> v2(v.size());
  foreach (i, v)
    v2[i] = v[i] / c;
  return v2;
}

template <typename T>
vector<T> operator * (const vector<T> &v, const T c) {
  vector<T> v2(v.size());
  foreach (i, v)
    v2[i] = v[i] * c;
  return v2;
}

template <typename T>
vector<T> operator + (const vector<T> &v1, const vector<T> &v2) {
  vector<T> sum(v1.size());
  std::transform(v1.begin(), v1.end(), v2.begin(), sum.begin(), std::plus<T>());
  return sum;
}

template <typename T>
vector<T> operator - (const vector<T> &v1, const vector<T> &v2) {
  vector<T> diff(v1.size());
  std::transform(v1.begin(), v1.end(), v2.begin(), diff.begin(), std::minus<T>());
  return diff;
}

template <typename T>
vector<vector<T> > split(const vector<T>& v, const vector<size_t>& lengths) {
  vector<vector<T> > result;

  size_t totalLength = 0;
  for (size_t i=0; i<lengths.size(); ++i)
    totalLength += lengths[i];

  assert(totalLength <= v.size());

  size_t offset = 0;
  for (size_t i=0; i<lengths.size(); ++i) {
    size_t l = lengths[i];
    vector<T> sub_v(l);
    for (size_t j=0; j<l; ++j)
      sub_v[j] = v[j+offset];

    result.push_back(sub_v);
    offset += l;
  }

  return result;
}

template <typename T> int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

std::string exec(std::string cmd);
namespace bash {
  vector<string> ls(string path);
}

string replace_all(const string& str, const string &token, const string &s);

#endif // _UTILITY_H_
