#include <ipc.h>

IPC::IPC(string fifo1, string fifo2) {
  _fifo1 = (fifo1.empty()) ? "/tmp/fifo1" : fifo1;
  _fifo2 = (fifo2.empty()) ? "/tmp/fifo2" : fifo2;

  __createFifoIfNotExists(_fifo1);
  __createFifoIfNotExists(_fifo2);
}

IPC::~IPC () {
  close(_in);
  close(_out);
}

template <typename T>
IPC& IPC::operator << (const vector<T>& vec) {
  cout << "Sending..." << endl;
  int _out = open(_fifo1.c_str(), O_WRONLY);

  size_t l = vec.size();
  size_t nBytesPerElement = sizeof(T);
  write(_out, &l, sizeof(l));
  write(_out, &nBytesPerElement, sizeof(nBytesPerElement));

  foreach (i, vec)
    write(_out, &vec[i], sizeof(vec[i]));

  close(_out);
  cout << "[Done]" << endl;
}

template <typename T>
IPC& IPC::operator >> (vector<T>& vec) {
  cout << "Receiving..." << endl;
  int _in  = open(_fifo2.c_str(), O_RDONLY);

  size_t length;
  while ( read(_in, &length, sizeof(length)) <= 0 );

  cout << "length = " << length << endl;

  size_t nBytesPerElement;
  read(_in, &nBytesPerElement, sizeof(nBytesPerElement));

  vec.resize(length);
  foreach (i, vec)
    read(_in, &vec[i], sizeof(T)); 
  close(_in);
  cout << "[Done]" << endl;

  return *this;
}

IPC& IPC::operator << (string msg) {
  int _out = open(_fifo1.c_str(), O_WRONLY);
  write(_out, msg.c_str(), msg.size());
  close(_out);

  return *this;
}

IPC& IPC::operator >> (string& ack) {
  const size_t MAX_BUF = 65536;
  char buf[MAX_BUF];

  int _in  = open(_fifo2.c_str(), O_RDONLY);
  do {
    read(_in, buf, MAX_BUF);
  } while ( string(buf).empty() );
  close(_in);

  ack = string(buf);
  return *this;
}

void IPC::clear_fifo() {
  exec("rm " + _fifo1);
  exec("rm " + _fifo2);
  __createFifoIfNotExists(_fifo1);
  __createFifoIfNotExists(_fifo2);
}

void IPC::__createFifoIfNotExists(string fifo) {
  string ret;
  if (!exists(fifo))
    ret = exec("mkfifo " + fifo);
  if (!ret.empty())
    throw "Cannot create fifo " + fifo;
}

template IPC& IPC::operator << (const vector<double>& vec);
template IPC& IPC::operator << (const vector<float>& vec);

template IPC& IPC::operator >> (vector<double>& vec);
template IPC& IPC::operator >> (vector<float>& vec);
