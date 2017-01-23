CXX=g++
CFLAGS=-I../mser/MSER/extrema-edu/extrema -I../mser/MSER/extrema-edu/imageutls -I../mser/MSER/extrema-edu/LL -I../mser/MSER/extrema-edu/optionGM -I../mser/MSER/extrema-edu/utls -I../mser/usr/local/include -I/usr/local/include/opencv2 -I../mser/ -I../misc/
LIBS=-L/usr/local/lib -L../mser/MSER/extrema-edu/extrema -L../mser/MSER/extrema-edu/LL -L../mser/MSER/extrema-edu/utls -L../mser/MSER/extrema-edu/optionGM

all:
	        $(CXX) -std=c++11 -g -o genDataset genDataset.cpp geometry.cpp ../mser/mser.cpp ../misc/utils.cpp $(CFLAGS) $(LIBS) -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem