#include <iostream>
#include <string>
#include <color.h>
#include <array.h>
#include <perf.h>

#include <model.h>

using namespace std;

GRADIENT& operator += (GRADIENT& g1, GRADIENT& g2);
void print(GRADIENT& g);
GRADIENT& calcGradient(Model& model, const vec& x, const vec& y);

void dnn_test();
void model_test();
void print(const std::vector<mat>& vm);

int main (int argc, char* argv[]) {

  model_test();
  dnn_test();

  return 0;
}

void dnn_test() {

  mat data("data/test.mx");
  mat target("data/test.mt");

  vector<size_t> dims(4);
  dims[0] = 15; dims[1] = 20; dims[2] = 30; dims[3] = 40;
  vector<mat> O(4);
  std::vector<mat> gradient;

  vector<float> coeff = ext::randn<float>(data.getRows());

  DNN dnn(dims);


  for (int itr=0; itr<16; ++itr) {
    dnn.feedForward(data, &O);

    print(O);

    mat error = target - O.back();
    range (i, error.getRows())
      range (j, error.getCols())
	error[i][j] = 0.5 * pow(error[i][j], 2.0);

    dnn.getEmptyGradient(gradient);
    dnn.backPropagate(error, O, gradient, coeff);
    dnn.updateParameters(gradient, 1e-4);
  }
}

void print(const std::vector<mat>& vm) {
  foreach (i, vm) {
    printf("rows = %lu, cols = %lu\n", vm[i].getRows(), vm[i].getCols());
    vm[i].print(3);
  }
}

void model_test() {

  vec x, y;
  ext::load(x, "data/test.vx");
  ext::load(y, "data/test.vy");

  // vec sigmoid_x = ext::b_sigmoid(x); ::print(x); ::print(sigmoid_x);
  // vec dsigma_x = dsigma(x); ::print(x, 6);   ::print(dsigma_x, 6);

  int M = 74;
  int WIDTH = 32;
  vector<size_t> d1(5), d2(5);
  d1[0] = 39; d1[1] = WIDTH; d1[2] = WIDTH + 1; d1[3] = WIDTH + 2; d1[4] = M;
  d2[0] =  M; d2[1] = WIDTH; d2[2] = WIDTH + 1; d2[3] = WIDTH + 2; d2[4] = 1;

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

  print(d);

  model.save("exp/dtwdnn.gpu/");
  
  printf("%f ms in total, avg %f ms per upate\n",
      timer.getTime(),
      timer.getTime() / MAX_ITR);

}


GRADIENT& calcGradient(Model& model, const vec& x, const vec& y) {
  model.evaluate(x, y); 
  model.calcGradient(x, y);
  return model.getGradient();
}

