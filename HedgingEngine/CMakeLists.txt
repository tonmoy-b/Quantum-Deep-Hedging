cmake_minimum_required(VERSION 3.10)
project(HedgingEngine LANGUAGES CXX)

# Standard C++ for high-performance math/concurrency
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add headers
include_directories(include)

# Find all source files in src/
file(GLOB SOURCES "src/*.cpp")

# Define the executable
add_executable(hedging_engine ${SOURCES})

# Platform-specific networking libraries (Winsock for Windows)
if(WIN32)
    target_link_libraries(hedging_engine ws2_32)
endif()

# Optimization for deep hedging workloads
target_compile_options(hedging_engine PRIVATE 
    $<$<CONFIG:Release>:-O3>
    $<$<PLATFORM_ID:Windows>:/W4>
    $<$<PLATFORM_ID:Linux>:-Wall -Wextra>
)