#ifndef __PHONE_STAT_H_
#define __PHONE_STAT_H_
#include <array.h>
#include <string>
#include <matrix.h>
#include <utility.h>
#include <trainable_dtw.h>
using namespace std;

void computeBetweenPhoneDistance(const Array<string>& phones, const string& MFCC_DIR, size_t N, const string& score_dir);
Matrix2D<double> comparePhoneDistances(const Array<string>& phones, const string& score_dir);
double average(const Matrix2D<double>& m, double MIN = -3.40e+34);
double objective(const Matrix2D<double>& scores);
Matrix2D<double> statistics (const Array<string>& phones);
void deduceCompetitivePhones(const Array<string>& phones, const Matrix2D<double>& scores);

void evaluate(bool reevaluate, const Array<string>& phones, string MFCC_DIR, size_t N, string matFile);

Array<string> getPhoneList(string filename);

#endif // __PHONE_STAT_H_
