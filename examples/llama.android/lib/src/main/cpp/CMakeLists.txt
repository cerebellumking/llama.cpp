cmake_minimum_required(VERSION 3.22.1)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# AI Chat library
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

set(HETEROSPEC_KLEIDIAI_ARCHIVE "" CACHE FILEPATH "Local KleidiAI release archive for offline Android Studio builds")
set(HETEROSPEC_KLEIDIAI_SOURCE_DIR "" CACHE PATH "Local extracted KleidiAI source directory for offline Android Studio builds")

if(GGML_CPU_KLEIDIAI)
    set(HETEROSPEC_DEFAULT_KLEIDIAI_ARCHIVE "${CMAKE_CURRENT_LIST_DIR}/third_party/kleidiai-v1.24.0-src.tar.gz")
    set(HETEROSPEC_KLEIDIAI_EXPECTED_MD5 "2f02ebe29573d45813e671eb304f2a00")

    if("${HETEROSPEC_KLEIDIAI_ARCHIVE}" STREQUAL "" AND EXISTS "${HETEROSPEC_DEFAULT_KLEIDIAI_ARCHIVE}")
        set(HETEROSPEC_KLEIDIAI_ARCHIVE "${HETEROSPEC_DEFAULT_KLEIDIAI_ARCHIVE}")
    endif()

    if(NOT "${HETEROSPEC_KLEIDIAI_SOURCE_DIR}" STREQUAL "")
        if(NOT IS_DIRECTORY "${HETEROSPEC_KLEIDIAI_SOURCE_DIR}/kai")
            message(FATAL_ERROR "HETEROSPEC_KLEIDIAI_SOURCE_DIR must point to an extracted KleidiAI source tree containing the kai directory: ${HETEROSPEC_KLEIDIAI_SOURCE_DIR}")
        endif()
        set(FETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD "${HETEROSPEC_KLEIDIAI_SOURCE_DIR}" CACHE PATH "Offline KleidiAI source directory" FORCE)
        message(STATUS "Using local KleidiAI source directory: ${HETEROSPEC_KLEIDIAI_SOURCE_DIR}")
    elseif(NOT "${HETEROSPEC_KLEIDIAI_ARCHIVE}" STREQUAL "")
        if(NOT EXISTS "${HETEROSPEC_KLEIDIAI_ARCHIVE}")
            message(FATAL_ERROR "HETEROSPEC_KLEIDIAI_ARCHIVE does not exist: ${HETEROSPEC_KLEIDIAI_ARCHIVE}")
        endif()

        file(MD5 "${HETEROSPEC_KLEIDIAI_ARCHIVE}" HETEROSPEC_KLEIDIAI_ARCHIVE_MD5)
        if(NOT "${HETEROSPEC_KLEIDIAI_ARCHIVE_MD5}" STREQUAL "${HETEROSPEC_KLEIDIAI_EXPECTED_MD5}")
            message(FATAL_ERROR "Local KleidiAI archive MD5 mismatch for ${HETEROSPEC_KLEIDIAI_ARCHIVE}. Expected ${HETEROSPEC_KLEIDIAI_EXPECTED_MD5}, got ${HETEROSPEC_KLEIDIAI_ARCHIVE_MD5}. Re-download kleidiai-v1.24.0-src.tar.gz.")
        endif()

        set(HETEROSPEC_KLEIDIAI_EXTRACT_ROOT "${CMAKE_BINARY_DIR}/_deps/kleidiai-local")
        set(HETEROSPEC_KLEIDIAI_MARKER "${HETEROSPEC_KLEIDIAI_EXTRACT_ROOT}/.archive-md5")
        set(HETEROSPEC_KLEIDIAI_SHOULD_EXTRACT TRUE)

        if(EXISTS "${HETEROSPEC_KLEIDIAI_MARKER}")
            file(READ "${HETEROSPEC_KLEIDIAI_MARKER}" HETEROSPEC_KLEIDIAI_CACHED_MD5)
            string(STRIP "${HETEROSPEC_KLEIDIAI_CACHED_MD5}" HETEROSPEC_KLEIDIAI_CACHED_MD5)
            if("${HETEROSPEC_KLEIDIAI_CACHED_MD5}" STREQUAL "${HETEROSPEC_KLEIDIAI_ARCHIVE_MD5}")
                set(HETEROSPEC_KLEIDIAI_SHOULD_EXTRACT FALSE)
            endif()
        endif()

        if(HETEROSPEC_KLEIDIAI_SHOULD_EXTRACT)
            file(REMOVE_RECURSE "${HETEROSPEC_KLEIDIAI_EXTRACT_ROOT}")
            file(MAKE_DIRECTORY "${HETEROSPEC_KLEIDIAI_EXTRACT_ROOT}")
            file(ARCHIVE_EXTRACT INPUT "${HETEROSPEC_KLEIDIAI_ARCHIVE}" DESTINATION "${HETEROSPEC_KLEIDIAI_EXTRACT_ROOT}")
            file(WRITE "${HETEROSPEC_KLEIDIAI_MARKER}" "${HETEROSPEC_KLEIDIAI_ARCHIVE_MD5}\n")
        endif()

        set(HETEROSPEC_KLEIDIAI_CANDIDATES "${HETEROSPEC_KLEIDIAI_EXTRACT_ROOT}")
        file(GLOB HETEROSPEC_KLEIDIAI_EXTRACTED_CHILDREN LIST_DIRECTORIES true "${HETEROSPEC_KLEIDIAI_EXTRACT_ROOT}/*")
        list(APPEND HETEROSPEC_KLEIDIAI_CANDIDATES ${HETEROSPEC_KLEIDIAI_EXTRACTED_CHILDREN})

        set(HETEROSPEC_KLEIDIAI_EXTRACTED_SOURCE_DIR "")
        foreach(HETEROSPEC_KLEIDIAI_CANDIDATE IN LISTS HETEROSPEC_KLEIDIAI_CANDIDATES)
            if(IS_DIRECTORY "${HETEROSPEC_KLEIDIAI_CANDIDATE}/kai")
                set(HETEROSPEC_KLEIDIAI_EXTRACTED_SOURCE_DIR "${HETEROSPEC_KLEIDIAI_CANDIDATE}")
                break()
            endif()
        endforeach()

        if("${HETEROSPEC_KLEIDIAI_EXTRACTED_SOURCE_DIR}" STREQUAL "")
            message(FATAL_ERROR "Could not find the KleidiAI kai directory after extracting ${HETEROSPEC_KLEIDIAI_ARCHIVE}")
        endif()

        set(FETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD "${HETEROSPEC_KLEIDIAI_EXTRACTED_SOURCE_DIR}" CACHE PATH "Offline KleidiAI source directory" FORCE)
        message(STATUS "Using local KleidiAI archive: ${HETEROSPEC_KLEIDIAI_ARCHIVE}")
        message(STATUS "Using extracted KleidiAI source directory: ${HETEROSPEC_KLEIDIAI_EXTRACTED_SOURCE_DIR}")
    endif()
endif()

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

if(NOT DEFINED OPENSSL_ROOT_DIR OR "${OPENSSL_ROOT_DIR}" STREQUAL "")
    set(OPENSSL_ROOT_DIR "$ENV{OPENSSL_ROOT_DIR}")
endif()
if(NOT DEFINED OPENSSL_ROOT_DIR OR "${OPENSSL_ROOT_DIR}" STREQUAL "")
    set(OPENSSL_ROOT_DIR "/Users/joe/Documents/软工课程/大四/科研/研一下/HeteroSpec/android_openssl/ssl_3/${ANDROID_ABI}")
endif()

add_compile_definitions(ASIO_STANDALONE)

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_chat.cpp
        llama_compat.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
        ${OPENSSL_ROOT_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/asio/include
        ${CMAKE_CURRENT_SOURCE_DIR}/websocketpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        llama-common
        android
        ${OPENSSL_ROOT_DIR}/libssl_3.so
        ${OPENSSL_ROOT_DIR}/libcrypto_3.so
        log)
