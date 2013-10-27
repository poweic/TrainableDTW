#ifndef __IPC_H
#define __IPC_H
#include <utility.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

class IPC {
  public:
    IPC(string fifo1 = "", string fifo2 = "");
    ~IPC ();

    IPC& operator << (string msg);
    IPC& operator >> (string& ack);

    template <typename T>
    IPC& operator << (const vector<T>& vec);

    template <typename T>
    IPC& operator >> (vector<T>& vec);

    string getFifo1() const { return _fifo1; }
    string getFifo2() const { return _fifo2; }

    void clear_fifo();
    
  private:
    void __createFifoIfNotExists(string fifo);

    string _fifo1;
    string _fifo2;
    int _out;
    int _in;
};

#endif // __IPC_H
