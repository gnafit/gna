cmake_minimum_required(VERSION 3.5)
project(wrap C CXX)

add_subdirectory(cuda/)

set(SRC wrapper.cxx)
cmake_policy(SET CMP0015 NEW)
link_directories(cuda/)

add_executable(wrapper ${SRC})

target_link_libraries(wrapper simp)
