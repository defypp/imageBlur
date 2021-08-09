set(CMAKE_BUILD_TYPE "Debug") 

include_directories(${OpenCV_INCLUDE_DIRS})

set(median
    ./median_demo.cpp
)

include_directories(../include)

add_executable(demo ${median})
target_link_libraries(demo 
                ${PROJECT_SOURCE_DIR}/bin/libfastMedianBlur.so 
                ${OpenCV_LIBS})


SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)       #设置可执行文件的输出目录
# exe path in debug and release
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin) 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin) 
