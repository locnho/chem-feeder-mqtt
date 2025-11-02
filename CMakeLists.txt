cmake_minimum_required(VERSION 3.5)   # Specify minimum CMake version
project(ph-dosing) 		      # Define project name and language
set(CMAKE_CXX_STANDARD 17)            # Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE) # Make C++ standard mandatory

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(chem-feeder-mqtt src/ph-chem-feeder.cpp src/mqtt-publish.cpp)
target_link_libraries(chem-feeder-mqtt ${OpenCV_LIBS} mosquitto tesseract leptonica)

add_executable(mqtt-sub src/mqtt-sub.cpp)
target_link_libraries(mqtt-sub mosquitto)

