VULCAN_ROOT=/share/Vulcan.v.0.12
UGOC_ROOT=/home/boton/Dropbox/DSP/ugoc/
RTK_UTIL_ROOT=/home/boton/Dropbox/DSP/RTK/utility

CC=gcc
CXX=g++-4.6 -Werror
CFLAGS=
NVCC=nvcc -arch=sm_21 -w

INCLUDE= -I include/ \
	 -I /usr/local/boton/include/ \
	 -I $(VULCAN_ROOT) \
	 -I $(UGOC_ROOT) \
	 -I $(UGOC_ROOT)/libsegtree/include \
	 -I $(UGOC_ROOT)/libfeature/include \
	 -I $(UGOC_ROOT)/libdtw/include \
	 -I $(UGOC_ROOT)/libutility/include \
	 -I /share/Local/ \
 	 -I /usr/local/cuda/samples/common/inc/ \
	 -I /usr/local/cuda/include \
	 -isystem $(VULCAN_ROOT)/am \
	 -isystem $(VULCAN_ROOT)/feature

CPPFLAGS= -std=c++0x -Wall -fstrict-aliasing $(CFLAGS) $(INCLUDE)

SOURCES=utility.cpp cdtw.cpp logarithmetics.cpp corpus.cpp archive_io.cpp blas.cpp model.cpp dnn.cpp #ipc.cpp 
EXAMPLE_PROGRAM=thrust_example dnn_example #ipc_example 
EXECUTABLES=train extract htk-to-kaldi kaldi-to-htk calc-acoustic-similarity pair-wise-dtw dtw-on-answer #$(EXAMPLE_PROGRAM) test 
 
.PHONY: debug all o3 example
all: $(EXECUTABLES) ctags

example: $(EXAMPLE_PROGRAM) ctags

o3: CFLAGS+=-O3
o3: all
debug: CFLAGS+=-g -DDEBUG
debug: all

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

OBJ=$(addprefix obj/,$(SOURCES:.cpp=.o))

LIBRARY= -lpbar \
	 -lprofile \
	 -larray \
	 -lmatrix \
	 $(VULCAN_ROOT)/am/vulcan-am.a\
	 $(VULCAN_ROOT)/feature/vulcan-feature.a\
	 $(VULCAN_ROOT)/common/vulcan-common.a\
	 -lgsl\
	 -lcblas\
	 -latlas\
	 -ldtw\
	 -lfeature\
	 -lsegtree\
	 -lutility

LIBRARY_PATH=-L/usr/local/boton/lib/ \
	     -Llib/\
	     -L$(UGOC_ROOT)/libdtw/lib/x86_64 \
	     -L$(UGOC_ROOT)/libfeature/lib/x86_64 \
	     -L$(UGOC_ROOT)/libsegtree/lib/x86_64 \
	     -L$(UGOC_ROOT)/libutility/lib/x86_64

#CU_OBJ=obj/dnn.o obj/device_matrix.o
#CU_LIB=-lcuda -lcublas

extract: $(OBJ) extract.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 

kaldi-to-htk: $(OBJ) kaldi-to-htk.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 
htk-to-kaldi: $(OBJ) htk-to-kaldi.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 

train: $(OBJ) train.cpp obj/trainable_dtw.o obj/phone_stat.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 
#train: $(OBJ) train.cu obj/trainable_dtw.o obj/phone_stat.o $(CU_OBJ)
#	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) $(CU_LIB)
test: $(OBJ) test.cpp 
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

calc-acoustic-similarity: $(OBJ) calc-acoustic-similarity.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

pair-wise-dtw: $(OBJ) pair-wise-dtw.cpp obj/fast_dtw.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
#pair-wise-dtw: $(OBJ) pair-wise-dtw.cpp obj/fast_dtw.o
#	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) $(CU_LIB)

dtw-on-answer: $(OBJ) dtw-on-answer.cpp obj/fast_dtw.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
#$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) $(CU_LIB)

ipc_example: $(OBJ) ipc_example.cpp ipc.h
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

dnn_example: $(OBJ) dnn_example.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

#dnn_example: $(OBJ) dnn_example.cu dnn.h $(CU_OBJ)

#thrust_example: $(OBJ) thrust_example.cu obj/device_matrix.o 
#$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) $(CU_LIB)
#-L$(RTK_UTIL_ROOT)/lib -lrtk 
ctags:
	@ctags -R *

obj/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -o $@ -c $<

obj/%.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.d: %.cpp
	@$(CXX) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$

-include $(addprefix obj/,$(subst .cpp,.d,$(SOURCES)))

.PHONY:
clean:
	rm -rf $(EXECUTABLES) $(EXAMPLE_PROGRAM) obj/*
