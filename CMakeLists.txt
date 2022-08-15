cmake_minimum_required (VERSION 3.16)

project (find_root)

set(CMAKE_CXX_STANDARD 17)



#if(NOT CMAKE_BUILD_TYPE)
  #set(CMAKE_BUILD_TYPE Release)
#endif()

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCES 
  src/main.cpp
  )

add_compile_options(-Wall)
add_executable(find_root ${SOURCES})


# Include googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  # commit for release-1.11.0
  URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)
FetchContent_MakeAvailable(googletest)

# Set the executable
enable_testing()

set(name_exe find_root-tests)
add_executable(${name_exe} tests/tests.cpp tests/performance.cpp)
target_link_libraries(${name_exe}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(${name_exe})
