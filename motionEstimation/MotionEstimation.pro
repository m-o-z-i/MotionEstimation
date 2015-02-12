TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++11 -fPIC -g -fexpensive-optimizations -D_GNULINUX -O3
QMAKE_EXTRA_TARGETS += libbaumer

SOURCES += \
    MotionEstimation.cpp \
    line/MyLine.cpp

HEADERS += \
    line/MyLine.h

unix:!macx: LIBS += -L/usr/local/lib/baumer \
                    -L/opt/boost/boost_1_55_0/lib \
                    -lbgapi2_ext \
                    -lbgapi2_genicam \
                    -lbgapi2_img \
                    -lopencv_core \
                    -lopencv_imgproc \
                    -lopencv_highgui \
                    -lopencv_calib3d \
                    -lopencv_contrib \
                    -lopencv_features2d \
                    -lopencv_objdetect \
                    -lopencv_video

INCLUDEPATH += $$PWD/../libbaumer/src/baumer/inc /opt/boost/boost_1_55_0/include
DEPENDPATH += $$PWD/../libbaumer/src/baumer/inc /opt/boost/boost_1_55_0/include
