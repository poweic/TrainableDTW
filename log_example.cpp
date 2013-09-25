#include <iostream>
#include "../include/logarithmetics.h"

using namespace std;

int main(int argc, const char *argv[]) {

  LLDouble z;
  cout << "z = " << z << endl;

  /* Initialize lin */
  LLDouble a(1e-307, LLDouble::LOGDOMAIN);
  cout << "a = " << a << endl;

  LLDouble b(1.1e-307, LLDouble::LOGDOMAIN);
  cout << "b = " << b << endl;

  cout << "a == b " << (a == b) << endl;
  cout << "a < b " << (a < b) << endl;
  cout << "a <= b " << (a <= b) << endl;
  cout << "a > b " << (a > b) << endl;
  cout << "a >= b " << (a >= b) << endl;

  b = LLDouble();
  cout << "b - a = " << b - a << endl;


  return 0;
}
