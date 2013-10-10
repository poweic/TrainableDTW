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
  vector<size_t> d1(4), d2(4);
  d1[0] = 39; d1[1] = WIDTH; d1[2] = WIDTH; d1[3] = M;
  d2[0] =  M; d2[1] = WIDTH; d2[2] = WIDTH; d2[3] = 1;

  Model model(d1, d2);

  const size_t MAX_ITR = 128;
  vec d(MAX_ITR);

  perf::Timer timer;
  timer.start();
  for (size_t i=0; i<MAX_ITR; ++i) {
    d[i] = model.evaluate(x, y);

    model.calcGradient(x, y);
    model.updateParameters(model.getGradient());
  }
  
  printf("%f ms in total, avg %f ms per upate\n", timer.getTime(), timer.getTime() / MAX_ITR);

  return 0;
}


GRADIENT& calcGradient(Model& model, const vec& x, const vec& y) {
  model.evaluate(x, y); 
  model.calcGradient(x, y);
  return model.getGradient();
}

