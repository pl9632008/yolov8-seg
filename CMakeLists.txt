cmake_minimum_required(VERSION 3.10)
project(headinferproject)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


set(SRC src/rail.cpp)
enable_language(CUDA)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS} /usr/local/cuda-11.4/include )
link_directories(/usr/local/cuda-11.4/lib64)

add_library(headinfer SHARED ${SRC})
target_link_libraries(headinfer nvinfer cudart ${OpenCV_LIBS})

add_executable(a ${SRC} src/main.cpp)
target_link_libraries(a nvinfer cudart ${OpenCV_LIBS})