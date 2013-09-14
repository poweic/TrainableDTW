#include <utility.h>

string int2str(int n) {
  char buf[32];
  sprintf(buf, "%d", n);
  return string(buf);
}

double str2double(const string& str) {
  return (double) atof(str.c_str());
}

float str2float(const string& str) {
  return atof(str.c_str());
}

int str2int(const string& str) {
  return atoi(str.c_str());
}

vector<string>& split(const string &s, char delim, vector<string>& elems) {
  stringstream ss(s);
  string item;
  while(getline(ss, item, delim))
    elems.push_back(item);
  return elems;
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  return split(s, delim, elems);
}

string replace_all(const string& str, const string &token, const string &s) {
  string result(str);
  size_t pos = 0;
  while((pos = result.find(token, pos)) != string::npos) {
    result.replace(pos, token.size(), s);
    pos += s.size();
  }
  return result;
}

bool isInt(string str) {
  int n = str2int(str);
  string s = int2str(n);

  return (s == str);
}

string join(const vector<string>& arr, string token) {
  string str;
  for (size_t i=0; i<arr.size() - 1; ++i)
      str += arr[i] + token;
  str += arr[arr.size() - 1];
  return str;
}
