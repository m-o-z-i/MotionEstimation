TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++11 -fPIC -g -fexpensive-optimizations -D_GNULINUX -O3
SOURCES += \
    MotionEstimation.cpp \
    FindCameraMatrices.cpp \
    FindPoints.cpp \
    Triangulation.cpp \
    Visualisation.cpp \
    PointCloudVis.cpp \
    main.cpp \
    Utility.cpp

HEADERS += \
    FindCameraMatrices.h \
    FindPoints.h \
    Triangulation.h \
    Visualisation.h \
    PointCloudVis.h \
    MotionEstimation.h \
    Utility.h

unix:!macx: LIBS += -lopencv_core \
                    -lopencv_imgproc \
                    -lopencv_highgui \
                    -lopencv_calib3d \
                    -lopencv_contrib \
                    -lopencv_features2d \
                    -lopencv_objdetect \
                    -lopencv_video \
                    -lpcl_visualization \
                    -lpcl_common \
                    -lpcl_filters \
                    -lvtkCommon \
                    -lvtkFiltering \
                    -lvtkRendering \
                    -lvtkGraphics \
                    -lboost_system

INCLUDEPATH += /usr/include/pcl-1.7 /usr/include/eigen3 /usr/include/vtk-5.8

