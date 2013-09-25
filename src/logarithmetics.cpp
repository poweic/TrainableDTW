#include <iostream>
#include <cassert>
#include <sstream>
#include "logarithmetics.h"

using std::stringstream;

/* min log domain operatable point */
const double LLDouble::LZERO   = -1.0E10;   /* ~log(0) */
const double LLDouble::LSMALL  = -0.5E10;
/* min log num when convert to linear (= log(MINLARG)) */
const double LLDouble::MINLOGARG = -708.3;
const double LLDouble::MINLINARG = 2.45E-308;


LLDouble operator+(const LLDouble& a, const LLDouble& b) {/*{{{*/
  assert(a._type == b._type);

  if (a._type == LLDouble::LINDOMAIN) {
    return LLDouble(a._val + b._val, LLDouble::LINDOMAIN);
  } else {
    double x = a._val, y = b._val, diff;
    if (x < y) std::swap(x, y); // make sure x > y
    diff = y - x; // diff < 0

    // if x >> y, return x
    if (diff < LLDouble::MINLOGARG) {
      return LLDouble(x);
      //return a;
    } else {
      return LLDouble(x + log(1.0 + exp(diff)), LLDouble::LOGDOMAIN);
    }
  }

}/*}}}*/

LLDouble operator-(const LLDouble& a, const LLDouble& b) /*{{{*/
  throw (LLDouble::MinusToNegExc) {
  assert(a._type == b._type);

  if (a._val < b._val) {
    stringstream err;
    err << "Error: " << a << " - " << b << " < 0.0";
    throw LLDouble::MinusToNegExc(err.str());
  }

  if (a._type == LLDouble::LINDOMAIN) {
    return LLDouble(a._val - b._val, LLDouble::LINDOMAIN);
  } else {
    double diff = b._val - a._val; // diff <= 0.0
    if (diff < LLDouble::MINLOGARG) {
      return a;
    } else {
      double z = a._val + LLDouble::LOG(1.0 - exp(diff));
      if (z < LLDouble::LSMALL) z = LLDouble::LZERO;
      return LLDouble(z, LLDouble::LOGDOMAIN);
    }
  }

}/*}}}*/

LLDouble operator*(const LLDouble& a, const LLDouble& b) {/*{{{*/
  assert(a._type == b._type);

  if (a._type == LLDouble::LINDOMAIN) {
    return LLDouble(a._val * b._val, LLDouble::LINDOMAIN);
  } else {
    double z = a._val + b._val;
    return (z <= LLDouble::LSMALL)
      ? LLDouble::LogZero()
      : LLDouble(z, LLDouble::LOGDOMAIN);
  }
}/*}}}*/

LLDouble operator/(const LLDouble& a, const LLDouble& b) /*{{{*/
  throw (LLDouble::DivideByZeroExc) {
  assert(a._type == b._type);

  if (a._type == LLDouble::LINDOMAIN) {
    if (b._val <= 0.0) {
      stringstream err;
      err << "Error: " << a << " / " << b << " divide by zero";
      throw LLDouble::DivideByZeroExc(err.str());
    }
    return LLDouble(a._val / b._val, LLDouble::LINDOMAIN);
  } else {
    if (b._val <= LLDouble::LSMALL) {
      stringstream err;
      err << "Error: " << a << " / " << b << " divide by zero";
      throw LLDouble::DivideByZeroExc(err.str());
    }
    double z = a._val - b._val;
    return (z <= LLDouble::LSMALL)
      ? LLDouble::LogZero()
      : LLDouble(z, LLDouble::LOGDOMAIN);
  }

}/*}}}*/

bool operator==(LLDouble a, LLDouble b) {/*{{{*/
  if (a._type == LLDouble::LINDOMAIN || b._type == LLDouble::LINDOMAIN) {
    a.to_lindomain();
    b.to_lindomain();
    return isEqual(a._val, b._val);
  } else {
    if (a._val < b._val) std::swap(a._val, b._val); // a >= b
    a -= b;
    return a._val < LLDouble::LSMALL;
  }
}/*}}}*/

bool operator<=(LLDouble a, LLDouble b) {/*{{{*/
  if (a._type == LLDouble::LINDOMAIN || b._type == LLDouble::LINDOMAIN) {
    a.to_lindomain();
    b.to_lindomain();
    return (a._val <= b._val) || isEqual(a._val, b._val);
  } else {
    return (a._val <= b._val) || a == b;
  }
}/*}}}*/

bool operator>=(LLDouble a, LLDouble b) {/*{{{*/
  if (a._type == LLDouble::LINDOMAIN || b._type == LLDouble::LINDOMAIN) {
    a.to_lindomain();
    b.to_lindomain();
    return (a._val >= b._val) || isEqual(a._val, b._val);
  } else {
    return (a._val >= b._val) || a == b;
  }
}/*}}}*/

bool operator<(LLDouble a, LLDouble b) {/*{{{*/
  return !(a >= b);
}/*}}}*/

bool operator>(LLDouble a, LLDouble b) {/*{{{*/
  return !(a <= b);
}/*}}}*/

std::ostream& operator<<(std::ostream& os, const LLDouble& ref) {/*{{{*/
  os << "LLDouble(" << ref._val << ", "
     << ((ref._type == LLDouble::LOGDOMAIN) ? "log)" : "lin)");
  return os;
}/*}}}*/



LLDouble::LLDouble(double d, const Type t) {/*{{{*/

  if (t == LINDOMAIN && d < LLDouble::MINLINARG) {
    d = 0.0;
  } else if (t == LOGDOMAIN && d < LSMALL) {
    d = LZERO;
  }
  _val = d;
  _type = t;
}/*}}}*/

LLDouble& LLDouble::to_logdomain() {/*{{{*/
  if (_type == LINDOMAIN) {
    _val = LOG(_val);
    _type = LOGDOMAIN;
  }
  return *this;
}/*}}}*/

LLDouble& LLDouble::to_lindomain() {/*{{{*/
  if (_type == LOGDOMAIN) {
    _val = EXP(_val);
    _type = LINDOMAIN;
  }
  return *this;
}/*}}}*/

const LLDouble& LLDouble::operator=(const LLDouble& ref) {/*{{{*/
  _val = ref._val;
  _type = ref._type;
  return ref;
}/*}}}*/

LLDouble& LLDouble::operator+=(const LLDouble& rhs) {/*{{{*/
  *this = *this + rhs;
  return *this;
}/*}}}*/

LLDouble& LLDouble::operator-=(const LLDouble& rhs) /*{{{*/
  throw (LLDouble::MinusToNegExc) {
  *this = *this - rhs;
  return *this;
}/*}}}*/

LLDouble& LLDouble::operator*=(const LLDouble& rhs) {/*{{{*/
  *this = *this * rhs;
  return *this;
}/*}}}*/

LLDouble& LLDouble::operator/=(const LLDouble& rhs) /*{{{*/
  throw (LLDouble::DivideByZeroExc) {
  *this = *this / rhs;
  return *this;
}/*}}}*/

bool LLDouble::iszero() const {/*{{{*/
  if (_type == LINDOMAIN) return _val < MINLINARG;
  else return _val < LSMALL;
}/*}}}*/



double LLDouble::LOG(double a) {/*{{{*/
  if (a < MINLINARG) return LZERO;
  else return log(a);
}/*}}}*/

double LLDouble::EXP(double a) /*{{{*/{
  if (a < MINLOGARG) return 0.0;
  else return exp(a);
}/*}}}*/


