CXX=g++ -std=c++11
CXXFLAGS:=-std=c++0x
INCS:=-I../caffe/distribute/include \
	  -I/usr/local/cuda/include

INCS+=$(shell pkg-config --cflags opencv)

LIBS:=-L/usr/lib/x86_64-linux-gnu/ -lprotobuf -lglog -lboost_system\
	  -L/usr/local/cuda/lib64/ -lcudart -lcublas -lcurand \
	  -L../caffe/build/lib/ -lcaffe
LIBS+=$(shell pkg-config --libs opencv)

all: detect.bin test_list.bin 
	@echo "MAKE ALL"

detect.bin: CaffePredictor.o Detector.o detect.o
	$(CXX) $(CXXFLAGS) $(INCS) -o detect.bin detect.o CaffePredictor.o Detector.o $(LIBS)

test_list.bin: test_list.o CaffePredictor.o
	$(CXX) $(CXXFLAGS) $(INCS) -o test_list.bin test_list.o CaffePredictor.o $(LIBS)

detect.o: detect.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c detect.cpp $(LIBS)

test_list.o: test_list.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c test_list.cpp $(LIBS)

Detector.o: Detector.hpp
	$(CXX) $(INCS) -c Detector.cpp $(LIBS)

CaffePredictor.o: CaffePredictor.hpp
	$(CXX) $(CXXFLAGS) $(INCS) -c CaffePredictor.cpp $(LIBS)

# classification:
# 	$(CXX) $(INCS) classification.cpp -o cln $(LIBS)

# check:
# 	@echo $(INCS)
# 	@echo $(LIBS)

clean:
	rm -rf *.o *.bin
