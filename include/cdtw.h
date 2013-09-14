#include <libutility/include/std_common.h>
#include <libutility/include/thread_util.h>
#include <libutility/include/utility.h>
#include <libdtw/include/dtw_parm.h>
#include <libdtw/include/dtw_util.h>

#define SOFT_POWER 4

inline double softmax(double x, double y, double z, double p = SOFT_POWER) {
  return pow( (pow(x, p) + pow(y, p) + pow(z, p)) / 3, 1/ p);
}

inline double softmin(double x, double y, double z, double p = SOFT_POWER) {
  // If more than one (>= 1) of the inputs are 0, the output would also be 0 (because 1/inf = 0.)
  return softmax(x, y, z, -p);
}

namespace DtwUtil {
  class CumulativeDtwRunner : public FrameDtwRunner {
    public:
      CumulativeDtwRunner(VectorDistFn norm) : FrameDtwRunner(norm) {}
      static unsigned nsnippet_;
    protected:
      virtual void CalScoreTable();
  };

};
