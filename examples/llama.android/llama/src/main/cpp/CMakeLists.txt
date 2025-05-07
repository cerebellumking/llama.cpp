# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("llama-android")

#include(FetchContent)
#FetchContent_Declare(
#        llama
#        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
#        GIT_TAG        master
#)

# Also provides "common"
#FetchContent_MakeAvailable(llama)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#

#load local llama.cpp
add_subdirectory(../../../../../../ build-llama)

set(OPENSSL_ROOT_DIR "")


add_definitions(-DASIO_STANDALONE)
## 添加WebSocket++和ASIO的本地路径
include_directories(
    ${OPENSSL_ROOT_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/asio/include
    ${CMAKE_CURRENT_SOURCE_DIR}/websocketpp
)

# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        llama-android.cpp)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        android
        ${OPENSSL_ROOT_DIR}/libssl_3.so
        ${OPENSSL_ROOT_DIR}/libcrypto_3.so
        log)

