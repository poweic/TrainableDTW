#include <iostream>
#include <string>
#include <color.h>
#include <array.h>
#include <ipc.h>

#include <cdtw.h>
#include <utility.h>
using namespace std;

int main (int argc, char* argv[]) {

  vector<double> t(39, 1.0);

  IPC ipc;
  ipc.clear_fifo();
  while (true) {
    ipc << t;
    ipc >> t;
    print(t);
  }

  return 0;

}
