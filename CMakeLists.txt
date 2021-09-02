cmake_minimum_required(VERSION 2.8)
project( opencv_practice )

# set(OpenCV_DIR /home/user/opencv/opencv3/build)
set(OpenCV_DIR /home/kevin/lib/opencv/opencv3/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( Eigen3 REQUIRED )
MESSAGE( [Main] " EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 
# EIGEN3_INCLUDE_DIRS = /usr/include/eigen3
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable( calibration src/opencv_sample/calibration.cpp )
target_link_libraries( calibration ${OpenCV_LIBS} )

add_executable( stereo_calib src/opencv_sample/stereo_calib.cpp )
target_link_libraries( stereo_calib ${OpenCV_LIBS} )

add_executable( stereo_match src/opencv_sample/stereo_match.cpp )
target_link_libraries( stereo_match ${OpenCV_LIBS} )