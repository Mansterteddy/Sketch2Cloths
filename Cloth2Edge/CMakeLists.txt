cmake_minimum_required(VERSION 3.8)
set( CMAKE_VERBOSE_MAKEFILE ON )

project(Cloth2Edge)

set(CMAKE_CXX_STANDARD 11)

get_filename_component( CINDER_PATH "/Users/manster/Documents/cinder_cmake/" ABSOLUTE )
get_filename_component( APP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" ABSOLUTE )

include( "${CINDER_PATH}/proj/cmake/modules/cinderMakeApp.cmake" )

include_directories(${CINDER_PATH}/blocks/Cinder-OpenCV-X/include)
LINK_DIRECTORIES(${CINDER_PATH}/blocks/Cinder-OpenCV-X/lib/osx)
LINK_LIBRARIES(libopencv_core.3.4.0.dylib)
LINK_LIBRARIES(libopencv_imgcodecs.3.4.0.dylib)
LINK_LIBRARIES(libopencv_imgproc.3.4.0.dylib)
LINK_LIBRARIES(libopencv_ximgproc.3.4.0.dylib)
LINK_LIBRARIES(libopencv_highgui.3.4.0.dylib)

ci_make_app(
        APP_NAME    "Cloth2Edge"
        SOURCES     ${APP_PATH}/src/main.cpp
        CINDER_PATH ${CINDER_PATH}
)