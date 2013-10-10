#include <iostream>
#include <string>
#include <color.h>
#include <array.h>
#include <perf.h>

#include <dnn.h>

using namespace std;

template <typename T>
vector<T>& operator += (vector<T>& x, vector<T>& y) {
  x = x + y;
  return x;
}

GRADIENT& operator += (GRADIENT& g1, GRADIENT& g2);
void print(GRADIENT& g);
GRADIENT& calcGradient(Model& model, const vec& x, const vec& y);

int main (int argc, char* argv[]) {
  vec x = loadvector("data/test.vx");
  vec y = loadvector("data/test.vy");

  int M = 74;
  int WIDTH = 40;
  Model model({39, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, M}, {M, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, 1});
  // auto g = calcGradient(model, x, y);

  const size_t MAX_ITR = 128;
  vec d(MAX_ITR);

  perf::Timer timer;
  timer.start();
  for (size_t i=0; i<MAX_ITR; ++i) {
    d[i] = model.evaluate(x, y);

    model.calcGradient(x, y);
    model.updateParameters(model.getGradient());
  }

  //print(d, 5);
  //print(blas::diff1st(d), 5);
  
  printf("%f ms in total, avg %f ms per upate\n", timer.getTime(), timer.getTime() / MAX_ITR);

  return 0;
}


GRADIENT& calcGradient(Model& model, const vec& x, const vec& y) {
  model.evaluate(x, y); 
  model.calcGradient(x, y);
  return model.getGradient();
}

