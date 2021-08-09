
#set(CMAKE_BUILD_TYPE "Debug") 
set(CMAKE_BUILD_TYPE "Release") 
set(SRC
    ./fast_median_blur.cpp
)

include_directories(../include)
add_library(fastMedianBlur SHARED ${SRC})



SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)           #设置库文件的输出目录

#lib path in debug and release
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)    
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin) 

