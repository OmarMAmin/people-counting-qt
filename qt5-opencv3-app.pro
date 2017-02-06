#############################################################################
##
## Makefile to integrate 3 key pieces of a typical system
##
##   1. Darknet/Yolo
##   2. Opencv3
##   3. Qt5
##
## Requires darknet-cpp port (atleast tag 3.75)
## https://github.com/prabindh/darknet/tree/v3.75
##
## Note: Darknet core is to be built as a shared lib (libdarknet-cpp-shared.so)
##
###############################################################################

QT += core
QT -= gui

CONFIG += c++11

TARGET = qt5_opencv3_darknet
CONFIG += console
CONFIG -= app_bundle

INCLUDEPATH += ../darknet-qt/src/

LIBS += \ 
    -L/usr/local/lib  \
    -lopencv_highgui -lopencv_videoio -lopencv_imgproc \
    -lopencv_photo -lopencv_core -lopencv_imgcodecs \
    -L./ -ldarknet-cpp-shared \
#-L/home/omaramin/Desktop/darknet-qt/




TEMPLATE = app

SOURCES += \
    main.cpp
