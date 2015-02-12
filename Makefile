# should be either OSC_HOST_BIG_ENDIAN or OSC_HOST_LITTLE_ENDIAN
# Apple: OSC_HOST_BIG_ENDIAN
# Win32: OSC_HOST_LITTLE_ENDIAN
# i386 LinuX: OSC_HOST_LITTLE_ENDIAN

PLATFORM=$(shell uname)

SDL_CFLAGS  := $(shell sdl-config --cflags)
SDL_LDFLAGS := $(shell sdl-config --libs) -lpthread

MOTION_ESTIMATION = MotionEstimation

INCLUDES = -I./libbaumer/src/baumer/inc -I/opt/boost/boost_1_55_0/include -Ipng++-0.2.3/ \
		   -I/opt/boost/boost_1_55_0/include -I./opencv-touch/src -I./line

LDPATH = -L./opencv-touch/src -L/opt/boost/boost_1_55_0/lib -L/usr/local/lib/baumer

LDFLAGS = -lbgapi2_ext \
          -lbgapi2_genicam \
          -lbgapi2_img \
          -lcamera_tools \
          -levisionlib \
          -limage_tools \
          -lMathParser \
          -lrt -lboost_thread -ljpeg -lpng -lGL -lglut -lpthread -lbgapi \
          -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_imgproc -lopencv_video -lboost_system \
          -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_objdetect

CFLAGS  = -fPIC -Wall -O3 $(SDL_CFLAGS) -DLINUX -D_GNULINUX
CXXFLAGS = $(CFLAGS) $(INCLUDES) -std=c++11

LINE_SOURCES = ./line/myline.cpp

MOTION_ESTIMATION_SOURCES = MotionEstimation.cpp
MOTION_ESTIMATION_OBJECTS = MotionEstimation.o

COMMON_SOURCES = $(LINE_SOURCES)
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)

all: obsticalavoidance opencv

obsticalavoidance: $(COMMON_OBJECTS) $(MOTION_ESTIMATION_OBJECTS)
	$(CXX) -o $(MOTION_ESTIMATION) $+ $(SDL_LDFLAGS) $(FRAMEWORKS) $(LDPATH) $(LDFLAGS)

opencv: $(COMMON_OBJECTS) $(MOTION_ESTIMATION_OBJECTS)
	cd ./opencv-touch/src/ && make

clean:
	rm -rf $(MOTION_ESTIMATION) $(MOTION_ESTIMATION_OBJECTS) $(COMMON_OBJECTS)
	cd ./opencv-touch/src/ && make clean
