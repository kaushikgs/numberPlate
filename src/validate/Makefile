CXX=g++
CFLAGS=-I./MSER/extrema-edu/extrema -I./MSER/extrema-edu/imageutls -I./MSER/extrema-edu/LL -I./MSER/extrema-edu/optionGM -I./MSER/extrema-edu/utls -I/usr/local/include -I/usr/local/include/opencv2
LIBS=-L./MSER/extrema-edu/extrema -L./MSER/extrema-edu/LL -L./MSER/extrema-edu/utls -L./MSER/extrema-edu/optionGM

all:
	        $(CXX) -std=c++11 -g -o validate validate.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_system -D USE_OPENCV -I ~/caffe/include -I ~/caffe/include/caffe/layers -L ~/caffe/build/lib -I /opt/OpenBLAS/include -L/opt/OpenBLAS/lib -I /usr/local/cuda/include -L /usr/local/cuda/lib64
