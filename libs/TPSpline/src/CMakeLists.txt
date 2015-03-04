cmake_minimum_required(VERSION 2.8)

project( TPSpline )

find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )

add_executable( TPSpline main.cpp CThinPlateSpline.cpp CThinPlateSpline.h )
target_link_libraries( TPSpline ${OpenCV_LIBS} )

