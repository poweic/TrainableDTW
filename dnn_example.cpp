#include <iostream>
#include <string>
#include <color.h>
#include <array.h>

#include <dnn.h>

using namespace std;

typedef Matrix2D<float> mat;

int main (int argc, char* argv[]) {
  vec x = loadvector("data/test.vx");
  vec y = loadvector("data/test.vy");

  int M = 4;
  Model model({39, 4, 4, 3, M}, {M, 4, 4, 3, 1});

  model.train(x, y);

  return 0;
}
