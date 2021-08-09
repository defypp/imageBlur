
cmake_minimum_required(VERSION 3.14)
set(OpenCV_DIR "/usr/local/share/OpenCV")
FIND_PACKAGE(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_definitions(-std=c++11)
# add_subdirectory(./Fast3_3MedianBlur)
add_subdirectory(./src)
add_subdirectory(./demo)