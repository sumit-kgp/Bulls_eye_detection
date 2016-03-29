#-------------------------------------------------
#
# Project created by QtCreator 2016-03-29T14:11:55
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = bulls_eye_detection
TEMPLATE = app
INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

SOURCES += main.cpp\
        mainwindow.cpp \
    calcmaxintensity.cpp \
    addstraightline.cpp \
    least_square_detection.cpp

HEADERS  += mainwindow.h \
    calcmaxintensity.h \
    addstraightline.h \
    least_square_detection.h

FORMS    += mainwindow.ui
