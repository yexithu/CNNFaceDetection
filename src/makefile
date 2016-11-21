CXX=g++
INCS:=-I../caffe/distribute/include \
	  -I/usr/local/cuda/include

INCS+=$(shell pkg-config --cflags opencv)

LIBS:=-L/usr/lib/x86_64-linux-gnu/ -lprotobuf -lglog -lboost_system\
	  -L/usr/local/cuda/lib64/ -lcudart -lcublas -lcurand \
	  -L../caffe/build/lib/ -lcaffe
LIBS+=$(shell pkg-config --libs opencv)

classification:
	$(CXX) $(INCS) classification.cpp -o cln $(LIBS)

check:
	@echo $(INCS)
	@echo $(LIBS)
