CXX=g++
CXXFLAGS:=-std=c++11
INCS:=-I../caffe/distribute/include \
	  -I/usr/local/cuda/include

INCS+=$(shell pkg-config --cflags opencv)

LIBS:=-L/usr/lib/x86_64-linux-gnu/ -lprotobuf -lglog -lboost_system\
	  -L/usr/local/cuda/lib64/ -lcudart -lcublas -lcurand \
	  -L../caffe/build/lib/ -lcaffe
LIBS+=$(shell pkg-config --libs opencv)


detect.bin: detect.o CaffePredictor.o
	$(CXX) $(CXXFLAGS) $(INCS) -o detect.bin detect.o CaffePredictor.o $(LIBS)

detect.o: detect.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c detect.cpp $(LIBS)

# Detector.o: Detector.hpp
# 	$(CXX) $(INCS) -c Detector.cpp $(LIBS)

CaffePredictor.o: CaffePredictor.hpp
	$(CXX) $(CXXFLAGS) $(INCS) -c CaffePredictor.cpp $(LIBS)

# classification:
# 	$(CXX) $(INCS) classification.cpp -o cln $(LIBS)

# check:
# 	@echo $(INCS)
# 	@echo $(LIBS)

clean:
	rm -rf *.o *.bin