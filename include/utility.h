#ifndef _UTILITY_H_
#define _UTILITY_H_
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cassert>
using namespace std;

string int2str(int n);
int str2int(const string& str);
float str2float(const string& str);
double str2double(const string& str);
string getValueStr(string& str);
string join(const vector<string>& arr);

bool isInt(string str);

vector<string> split(const string &s, char delim);
vector<string>& split(const string &s, char delim, vector<string>& elems);

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




string replace_all(const string& str, const string &token, const string &s);
#endif // _UTILITY_H_
