#include <iostream>
#include <string>
#include <color.h>
#include <array.h>

#include <cdtw.h>
using namespace std;

#ifdef DEBUG
#define debug(token) {cout << #token " = " << token << endl;}
#else
#define debug(token) {}
#endif

#define function []

template<typename T>
int test(T&& callback) {
  int x  = 10;
  std::forward<T>(callback)(x);
}

string bool2str(bool b);

template<typename T>
int DTW(int x, int y, T&& fn) {
  return std::forward<T>(fn)(x, y);
}

int main (int argc, char* argv[]) {

  Array<string> full("full");
  Array<string> done("done");

  int counter = 0;
  foreach (i, full) {
    if (!exists("/share/preparsed_files/SI_word_test/mul-sim/" + full[i] + ".mul-sim")) {
      cout << full[i] << endl;
      ++counter;
    }
  }

  cout << "counter = " << counter << endl;

  return 0;

  int m = 10, n = 20;
  int diag = -1;
  int d = DTW(m, n, [=] (int x, int y) {
    return x*y*diag;
  });

  debug(d);

  auto f = [](int n) {
    return [=](int x) mutable {
      return pow(x, n);
    };
  };

  test(function (int x) { 
      cout << x << endl;
      });

  auto f1 = f(5);
  auto f2 = f(10);
  debug(f1(2));
  debug(f2(2));

  int arr[5] = {1, 2, 3, 4, 5};
  for (auto &x: arr)
    debug(x);

  return 0;

  LLDouble x(1, LLDouble::LOGDOMAIN);
  LLDouble y(2, LLDouble::LOGDOMAIN);
  LLDouble z(3, LLDouble::LOGDOMAIN);

  // sum will be 3.40760596444438 in log-domain
  LLDouble sum = x + y + z;

  cout << "----------- "GREEN"LOG Domain"COLOREND" -----------" << endl;
  cout << "  sum          = " << sum << endl;
  cout << "  sum.getVal() = " << sum.getVal() << endl;
  cout << "----------- "GREEN"LIN Domain"COLOREND" -----------" << endl;
  cout << "  sum          = " << sum.to_lindomain() << endl;
  cout << "  sum.getVal() = " << sum.getVal() << endl << endl;

  double a=1, b=5, c=5;
  double min = SMIN::eval(a, b, c);
  cout << "-------- "GREEN"Smoothed Minimum"COLOREND" --------" << endl;
  cout << " smoothed min  = " << min << endl << endl;

  /*
  LLDouble z;
  cout << "z = " << z << endl;

  // Initialize lin
  LLDouble a(1e-307, LLDouble::LOGDOMAIN);
  cout << "a = " << a << endl;

  LLDouble b(1.1e-307, LLDouble::LOGDOMAIN);
  cout << "b = " << b << endl;

  cout << "a == b " << bool2str(a == b) << endl;
  cout << "a <  b " << bool2str(a <  b) << endl;
  cout << "a <= b " << bool2str(a <= b) << endl;
  cout << "a >  b " << bool2str(a >  b) << endl;
  cout << "a >= b " << bool2str(a >= b) << endl;

  b = LLDouble();
  try {
    cout << "b - a = " << b - a << endl;
  }
  catch (...) {
    cerr << RED "[Exception]" COLOREND " Caught in " << __FUNCTION__ << " at line: " << __LINE__ << ORANGE " <== " COLOREND " Nagative number in linear domain CANNOT be mapped into LOG-domain. (i.e. log(-10) = ?? IT'S a IMAGINARY number )" << endl;
  }
  */


  return 0;
}

string bool2str(bool b) {
  return b ? "true" : "false";
}
