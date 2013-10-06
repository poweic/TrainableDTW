#include <iostream>
#include <string>
#include <color.h>
#include <array.h>

#include <dnn.h>

using namespace std;

template <typename T>
vector<T>& operator += (vector<T>& x, vector<T>& y) {
  x = x + y;
  return x;
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
  model.calcGradients(x, y);
  return model.getGradient();
}

int main (int argc, char* argv[]) {
  vec x = loadvector("data/test.vx");
  vec y = loadvector("data/test.vy");

  int M = 74;
  Model model({39, 4, 4, 3, M}, {M, 4, 4, 3, 1});
  //model.train(x, y);
  
  auto g = calcGradient(model, x, y);

  for (size_t i=0; i<128; ++i)
    g += calcGradient(model, x, y);

  return 0;
}
