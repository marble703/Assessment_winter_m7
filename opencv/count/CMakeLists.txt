CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

project(ccamera)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_DIRS}
)

add_executable(cv main.cpp)

target_link_libraries(cv ${OpenCV_LIBS})