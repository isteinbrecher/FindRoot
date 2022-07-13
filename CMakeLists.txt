cmake_minimum_required (VERSION 3.16)

project (newton)

set(CMAKE_CXX_STANDARD 17)



#if(NOT CMAKE_BUILD_TYPE)
  #set(CMAKE_BUILD_TYPE Release)
#endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCES 
  src/main.cpp
  )

add_compile_options(-Wall)
add_executable(nr ${SOURCES})



# # CMakeLists.txt

# ...
# add_library(foo ...)
# ...
