target = nn1.exe

src = nn1.cpp

CXXFLAG := -std=gnu++14 -D_GLIBCXX_USE_CXX11_ABI=0

INC = -I/opt/libtorch/include/torch/csrc/api/include -I/opt/libtorch/include

DL = -L/opt/libtorch -ltorch -ltorch_cpu -lc10

CXX = clang++

default: all
all : $(target)

$(target): $(src)
	$(CXX) $< $(CXXFLAG) $(INC) $(DL) -o $@

debug: CXXFLAG += -DDEBUG -g -pg 
debug: $(target)

clean:
	rm *.exe
