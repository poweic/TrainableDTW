#include <iostream>
#include <string>
#include <color.h>
#include <array.h>
#include <profile.h>

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
float evaluate(Model& model, const vec& x, const vec& y);

int main (int argc, char* argv[]) {
  vec x = loadvector("data/test.vx");
  vec y = loadvector("data/test.vy");

  int M = 74;
  int WIDTH = 4096;
  Model model({39, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, M}, {M, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, 1});
  // auto g = calcGradient(model, x, y);

  const size_t MAX_ITR = 128;
  vec d(MAX_ITR);

  util::Timer timer;
  timer.start();
  for (size_t i=0; i<MAX_ITR; ++i) {
    d[i] = evaluate(model, x, y);

    model.calcGradient(x, y);
    model.updateParameters(model.getGradient());
  }

  print(d, 5);
  print(blas::diff1st(d), 5);
  
  printf("%f ms in total, avg %f ms per upate\n", timer.getTime(), timer.getTime() / MAX_ITR);

  return 0;
}

GRADIENT& operator += (GRADIENT& g1, GRADIENT& g2) {
  GRADIENT_ALIASING(g1, g1_1, g1_2, g1_3, g1_4);
  GRADIENT_ALIASING(g2, g2_1, g2_2, g2_3, g2_4);

  foreach (i, g1_1)
    g1_1[i] += g2_1[i];

  foreach (i, g1_2)
    g1_2[i] += g2_2[i];

  g1_3 += g2_3;

  foreach (i, g1_4)
    g1_4[i] += g2_4[i];

  return g1;
}

void print(GRADIENT& g) {
  GRADIENT_ALIASING(g, g1, g2, g3, g4);
  
  foreach (i, g1)
    g1[i].print();

  foreach (i, g2)
    g2[i].print();

  cout << endl;
  print(g3);

  foreach (i, g4)
    g4[i].print();
}

GRADIENT& calcGradient(Model& model, const vec& x, const vec& y) {
  model.evaluate(x, y); 
  model.calcGradient(x, y);
  return model.getGradient();
}

float evaluate(Model& model, const vec& x, const vec& y) {
  model.evaluate(x, y);
  auto& o = std::get<3>(model.getHiddenOutput());
  auto d = o[o.size() - 1][0];
  return d;
}

