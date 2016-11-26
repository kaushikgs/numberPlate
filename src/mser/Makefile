CXX=g++
CFLAGS=-I./MSER/extrema-edu/extrema -I./MSER/extrema-edu/imageutls -I./MSER/extrema-edu/LL -I./MSER/extrema-edu/optionGM -I./MSER/extrema-edu/utls -I/usr/local/include -I/usr/local/include/opencv2
LIBS=-L/usr/local/lib -L./MSER/extrema-edu/extrema -L./MSER/extrema-edu/LL -L./MSER/extrema-edu/utls -L./MSER/extrema-edu/optionGM

all:
	        $(CXX) -std=c++11 -g -c mser.cpp $(CFLAGS) $(LIBS) -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem
