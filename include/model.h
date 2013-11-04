#ifndef __MODEL_H_
#define __MODEL_H_

#include <dnn.h>

class Model {
public:

  Model();
  Model(const std::vector<size_t>& pp_dim, const std::vector<size_t>& dtw_dim);
  Model(const Model& src);
  Model& operator = (Model rhs);

  void load(string folder);
  void initHiddenOutputAndGradient();

  void train(const vec& x, const vec& y);
  float evaluate(const vec& x, const vec& y);
  float evaluate(const float* x, const float* y);
  void calcGradient(const vec& x, const vec& y);
  void calcGradient(const float* x, const float* y);
  void updateParameters(GRADIENT& g);
  void setLearningRate(float learning_rate);

  HIDDEN_OUTPUT& getHiddenOutput();
  GRADIENT& getGradient();
  void getEmptyGradient(GRADIENT& g);
  void save(string folder) const;
  void print() const;

  friend void swap(Model& lhs, Model& rhs);
 
private:
  HIDDEN_OUTPUT hidden_output;
  GRADIENT gradient;

  float _lr;

  vec _w;
  DNN _pp;
  DNN _dtw;
};

void swap(Model& lhs, Model& rhs);

GRADIENT& operator += (GRADIENT& g1, const GRADIENT& g2);
GRADIENT& operator -= (GRADIENT& g1, const GRADIENT& g2);
GRADIENT& operator *= (GRADIENT& g, float c);
GRADIENT& operator /= (GRADIENT& g, float c);

GRADIENT operator + (GRADIENT g1, const GRADIENT& g2);
GRADIENT operator - (GRADIENT g1, const GRADIENT& g2);
GRADIENT operator * (GRADIENT g, float c);
GRADIENT operator * (float c, GRADIENT g);
GRADIENT operator / (GRADIENT g, float c);

//bool hasNAN(GRADIENT& g);
void print(GRADIENT& g);
//float sum(GRADIENT& g);


#endif // __MODEL_H_
