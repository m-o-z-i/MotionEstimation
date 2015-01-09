# should be either OSC_HOST_BIG_ENDIAN or OSC_HOST_LITTLE_ENDIAN
# Apple: OSC_HOST_BIG_ENDIAN
# Win32: OSC_HOST_LITTLE_ENDIAN
# i386 LinuX: OSC_HOST_LITTLE_ENDIAN

PLATFORM=$(shell uname)

SDL_CFLAGS  := $(shell sdl-config --cflags)
SDL_LDFLAGS := $(shell sdl-config --libs) -lpthread

OBSTICAL_AVOIDANCE = ObsticalAvoidance

INCLUDES = -I./libbaumer/src/baumer/inc -I/opt/boost/boost_1_55_0/include -Ipng++-0.2.3/ \
		   -I/opt/boost/boost_1_55_0/include -I./opencv-touch/src

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

OBSTICAL_AVOIDANCE_SOURCES = ObsticalAvoidance.cpp
OBSTICAL_AVOIDANCE_OBJECTS = ObsticalAvoidance.o

all: obsticalavoidance opencv

obsticalavoidance: $(OBSTICAL_AVOIDANCE_OBJECTS)
	$(CXX) -o $(OBSTICAL_AVOIDANCE) $+ $(SDL_LDFLAGS) $(FRAMEWORKS) $(LDPATH) $(LDFLAGS)

opencv: $(OBSTICAL_AVOIDANCE_OBJECTS)
	cd ./opencv-touch/src/ && make

clean:
	rm -rf $(OBSTICAL_AVOIDANCE) $(OBSTICAL_AVOIDANCE_OBJECTS)
	cd ./opencv-touch/src/ && make clean
