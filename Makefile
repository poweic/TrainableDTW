VULCAN_ROOT=/media/Data1/Vulcan.v.0.12
UGOC_ROOT=/home/boton/Dropbox/DSP/ugoc/
RTK_UTIL_ROOT=/home/boton/Dropbox/DSP/RTK/utility

CC=gcc
CXX=g++
CFLAGS=

INCLUDE= -I include/ \
	 -I /usr/local/boton/include/ \
	 -I $(VULCAN_ROOT)/am \
	 -I $(VULCAN_ROOT)/feature \
	 -I $(VULCAN_ROOT) \
	 -I $(UGOC_ROOT) \
	 -I $(UGOC_ROOT)/libsegtree/include \
	 -I $(UGOC_ROOT)/libfeature/include \
	 -I $(UGOC_ROOT)/libdtw/include \
	 -I $(UGOC_ROOT)/libutility/include

#-I $(RTK_UTIL_ROOT)/include
#-I $(UGOC_ROOT)/libfeature/include \
#-I $(UGOC_ROOT)/libutility/include \

CPPFLAGS=-std=c++0x -w -fstrict-aliasing $(CFLAGS) $(INCLUDE)

SOURCES=utility.cpp cdtw.cpp logarithmetics.cpp corpus.cpp
EXECUTABLES=extract train test calc-acoustic-similarity
 
.PHONY: debug all o3
all: $(EXECUTABLES) ctags

o3: CFLAGS+=-O3
o3: all
debug: CFLAGS+=-g -DDEBUG
debug: all

vpath %.h include/
vpath %.cpp src/

OBJ=$(addprefix obj/,$(SOURCES:.cpp=.o))

LIBRARY= -lcmdparser \
	 -lpbar \
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

extract: $(OBJ) extract.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY) 
train: $(OBJ) train.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
test: $(OBJ) test.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
calc-acoustic-similarity: $(OBJ) calc-acoustic-similarity.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
#-L$(RTK_UTIL_ROOT)/lib -lrtk 
ctags:
	@ctags -R *

obj/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -o $@ -c $<

obj/%.d: %.cpp
	@$(CXX) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$

-include $(addprefix obj/,$(subst .cpp,.d,$(SOURCES)))

.PHONY:
clean:
	rm -rf $(EXECUTABLES) obj/*
