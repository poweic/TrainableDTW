#include <iostream>
#include <string>
#include <color.h>
#include <array.h>
#include <perf.h>

#include <model.h>

using namespace std;

// typedef device_matrix<float> mat;
// typedef thrust::device_vector<float> vec;

GRADIENT& operator += (GRADIENT& g1, GRADIENT& g2);
void print(GRADIENT& g);
GRADIENT& calcGradient(Model& model, const vec& x, const vec& y);

Model model;

int main (int argc, char* argv[]) {
  vec x, y;
  ext::load(x, "data/test.vx");
  ext::load(y, "data/test.vy");

  //vec sigmoid_x = ext::b_sigmoid(x); ::print(x); ::print(sigmoid_x);
  vec dsigma_x = dsigma(x); ::print(x, 6);   ::print(dsigma_x, 6);

  int M = 74;
  int WIDTH = 32;
  vector<size_t> d1(5), d2(5);
  d1[0] = 39; d1[1] = WIDTH; d1[2] = WIDTH; d1[3] = WIDTH; d1[4] = M;
  d2[0] =  M; d2[1] = WIDTH; d2[2] = WIDTH; d2[3] = WIDTH; d2[4] = 1;

  model = Model(d1, d2);

  const size_t MAX_ITR = 128;
  vec d(MAX_ITR);

  perf::Timer timer;
  timer.start();
  for (size_t i=0; i<MAX_ITR; ++i) {
    d[i] = model.evaluate(x, y);

    model.calcGradient(x, y);
    model.updateParameters(model.getGradient());
  }

  print(d);

  model.save("exp/dtwdnn.gpu/");
  
  printf("%f ms in total, avg %f ms per upate\n", timer.getTime(), timer.getTime() / MAX_ITR);

  return 0;
}


GRADIENT& calcGradient(Model& model, const vec& x, const vec& y) {
  model.evaluate(x, y); 
  model.calcGradient(x, y);
  return model.getGradient();
}

