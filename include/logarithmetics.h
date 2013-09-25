#ifndef __LOGARITHMETICS_H__
#define __LOGARITHMETICS_H__

#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <string>

using std::string;


/*                LIN      |      LOG
 * ------------------------+-----------------------
 * 2.45e-308 = MINLINARG <-+-> MINLOGARG = -708.3
 *         0 =           <-+-> LSMALL    = -0.5e10
 *         0 =           <-+-> LZERO     = -1e10
 * ------------------------+-----------------------
 * Log domain operation range: LSMALL ~ inf
 * If Log -> Lin, operation range: MINLOGARG ~ inf
 */
class LLDouble {
  public:
    /* Exceptions */
    class MinusToNegExc : public std::exception {
      public:
        MinusToNegExc(string s) { errmsg = s; }
        ~MinusToNegExc() throw () {}
        virtual const char* what() const throw() { return errmsg.c_str(); }
      protected:
        string errmsg;
    };

    class DivideByZeroExc: public std::exception {
      public:
        DivideByZeroExc(string s) { errmsg = s; }
        ~DivideByZeroExc() throw () {}
        virtual const char* what() const throw() { return errmsg.c_str(); }
      protected:
        string errmsg;
    };

    friend LLDouble operator + (const LLDouble& a, const LLDouble& b);
    friend LLDouble operator - (const LLDouble& a, const LLDouble& b) throw (MinusToNegExc);
    friend LLDouble operator * (const LLDouble& a, const LLDouble& b);
    friend LLDouble operator / (const LLDouble& a, const LLDouble& b) throw (DivideByZeroExc);

    friend bool operator == (LLDouble a, LLDouble b);
    friend bool operator <  (LLDouble a, LLDouble b);
    friend bool operator <= (LLDouble a, LLDouble b);
    friend bool operator >  (LLDouble a, LLDouble b);
    friend bool operator >= (LLDouble a, LLDouble b);
    friend std::ostream& operator<<(std::ostream& os, const LLDouble& ref);

    enum Type {LOGDOMAIN, LINDOMAIN};
    LLDouble() : _val(LZERO), _type(LOGDOMAIN) {}
    LLDouble(const LLDouble& ref) { *this = ref; }
    LLDouble(const double d, const Type t = LOGDOMAIN);

    LLDouble& to_logdomain();
    LLDouble& to_lindomain();

    const LLDouble& operator=(const LLDouble& ref);

    LLDouble& operator += (const LLDouble& ref);
    LLDouble& operator -= (const LLDouble& ref) throw (MinusToNegExc);
    LLDouble& operator *= (const LLDouble& ref);
    LLDouble& operator /= (const LLDouble& ref) throw (DivideByZeroExc);
    bool iszero() const;

    // Default gives Log zero
    static LLDouble LogZero() { return LLDouble(); }

    double getVal() const { return _val; }

  private:
    static const double LZERO;    /* ~log(0) */
    static const double LSMALL;
    static const double MINLOGARG;  /* lowest exp() arg  = log(MINLARG) */
    static const double MINLINARG;  /* lowest log() arg  = exp(MINEARG) */

    static double LOG(double a);
    static double EXP(double a);

    double _val;
    Type _type;
};


inline bool isEqual(double a, double b, double e = 0.0) {
  if (e == 0.0) {
    e = std::min<double>(fabs(a), fabs(b))
      * std::numeric_limits<double>::epsilon();
  }
  return fabs(a - b) <=  e ;
}

#endif /* __LOGARITHMETICS_H__ */
