VULCAN_ROOT=/media/Data1/Vulcan.v.0.12
UGOC_ROOT=/home/boton/Dropbox/DSP/ugoc/
RTK_UTIL_ROOT=/home/boton/Dropbox/DSP/RTK/utility

CC=gcc
CXX=g++
CFLAGS=
NVCC=nvcc -arch=sm_21 -w

INCLUDE= -I include/ \
	 -I /usr/local/boton/include/ \
	 -I $(VULCAN_ROOT)/am \
	 -I $(VULCAN_ROOT)/feature \
	 -I $(VULCAN_ROOT) \
	 -I $(UGOC_ROOT) \
	 -I $(UGOC_ROOT)/libsegtree/include \
	 -I $(UGOC_ROOT)/libfeature/include \
	 -I $(UGOC_ROOT)/libdtw/include \
	 -I $(UGOC_ROOT)/libutility/include \
	 -I /share/Local/ \
 	 -I /usr/local/cuda/samples/common/inc/ \
	 -I /usr/local/cuda/include

#-I $(RTK_UTIL_ROOT)/include
#-I $(UGOC_ROOT)/libfeature/include \
#-I $(UGOC_ROOT)/libutility/include \

CPPFLAGS= -std=c++0x -w -fstrict-aliasing $(CFLAGS) $(INCLUDE)

SOURCES=utility.cpp fast_dtw.cpp cdtw.cpp logarithmetics.cpp corpus.cpp ipc.cpp archive_io.cpp blas.cpp 
EXAMPLE_PROGRAM=thrust_example ipc_example dnn_example
EXECUTABLES=train extract kaldi-to-htk calc-acoustic-similarity calc-acoustic-similarity-fast #$(EXAMPLE_PROGRAM) test 
 
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
	 -ldtw\
	 -lfeature\
	 -lsegtree\
	 -lutility
#-lrtk\

LIBRARY_PATH=-L/usr/local/boton/lib/ \
	     -Llib/\
	     -L$(UGOC_ROOT)/libdtw/lib/x86_64 \
	     -L$(UGOC_ROOT)/libfeature/lib/x86_64 \
	     -L$(UGOC_ROOT)/libsegtree/lib/x86_64 \
	     -L$(UGOC_ROOT)/libutility/lib/x86_64

#-L$(RTK_UTIL_ROOT)/lib
CU_OBJ=obj/dnn.o obj/device_matrix.o
CU_LIB=-lcuda -lcublas

extract: $(OBJ) extract.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 
kaldi-to-htk: $(OBJ) kaldi-to-htk.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 
#train: $(OBJ) train.cpp obj/trainable_dtw.o obj/phone_stat.o
	#$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 
train: $(OBJ) train.cu obj/trainable_dtw.o obj/phone_stat.o $(CU_OBJ)
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) $(CU_LIB)
test: $(OBJ) test.cpp 
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

#calc-acoustic-similarity: $(OBJ) calc-acoustic-similarity.cpp obj/trainable_dtw.o 
#$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

calc-acoustic-similarity: $(OBJ) calc-acoustic-similarity.cu obj/trainable_dtw.o 
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) $(CU_OBJ) $(CU_LIB)
calc-acoustic-similarity-fast: $(OBJ) calc-acoustic-similarity-fast.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 

ipc_example: $(OBJ) ipc_example.cpp ipc.h
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)

#dnn_example: $(OBJ) dnn_example.cu dnn.h $(CU_OBJ)
#$(NVCC) -w $(CFLAGS) $(INCLUDE) -o $@ $< $(CU_OBJ) $@.cu $(LIBRARY_PATH) $(LIBRARY) $(CU_LIB)

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
