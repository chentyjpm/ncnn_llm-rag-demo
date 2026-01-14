cmake_minimum_required(VERSION 3.16)

project(ncnn_llm_rag_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
include(CheckCXXSourceCompiles)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_Declare(
  cpp_httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.14.0.tar.gz
)
FetchContent_MakeAvailable(nlohmann_json cpp_httplib)

set(NCNN_TARGET "")
# Prefer explicit paths (works best for CI with prebuilt zips).
if (NCNN_LIBRARY AND NCNN_INCLUDE_DIR)
  get_filename_component(_ncnn_include_dir_abs "${NCNN_INCLUDE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  get_filename_component(_ncnn_library_abs "${NCNN_LIBRARY}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  add_library(ncnn_manual UNKNOWN IMPORTED)
  set_target_properties(ncnn_manual PROPERTIES
    IMPORTED_LOCATION "${_ncnn_library_abs}"
    INTERFACE_INCLUDE_DIRECTORIES "${_ncnn_include_dir_abs}"
  )
  set(NCNN_TARGET ncnn_manual)
else()
  find_package(ncnn QUIET)
  if (TARGET ncnn::ncnn)
    set(NCNN_TARGET ncnn::ncnn)
  elseif (TARGET ncnn)
    set(NCNN_TARGET ncnn)
  endif()

  if (NOT NCNN_TARGET)
    find_path(NCNN_INCLUDE_DIR ncnn/net.h)
    find_library(NCNN_LIBRARY ncnn)
    if (NOT NCNN_INCLUDE_DIR OR NOT NCNN_LIBRARY)
      message(FATAL_ERROR "ncnn not found. Set NCNN_INCLUDE_DIR and NCNN_LIBRARY, or install ncnn with CMake package files.")
    endif()
    add_library(ncnn UNKNOWN IMPORTED)
    set_target_properties(ncnn PROPERTIES
      IMPORTED_LOCATION "${NCNN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NCNN_INCLUDE_DIR}"
    )
    set(NCNN_TARGET ncnn)
  endif()
endif()

find_package(Threads REQUIRED)

add_executable(ncnn_llm_rag_app
  src/app_main.cpp
  src/rag_vector_db.cpp
  src/rag_text.cpp
  src/rag_ingest.cpp
  third_party/sqlite/sqlite-amalgamation-3510200/sqlite3.c
  ncnn_llm/src/ncnn_llm_gpt.cpp
  ncnn_llm/src/utils/rope_embed.cpp
  ncnn_llm/src/utils/prompt.cpp
  ncnn_llm/src/utils/tokenizer/bpe_tokenizer.cpp
  ncnn_llm/src/utils/tokenizer/unigram_tokenizer.cpp
  ncnn_llm/examples/llm_ncnn_run/json_utils.cpp
  ncnn_llm/examples/llm_ncnn_run/util.cpp
)

target_include_directories(ncnn_llm_rag_app PRIVATE
  src
  third_party/sqlite/sqlite-amalgamation-3510200
  ncnn_llm/src
  ncnn_llm/examples/llm_ncnn_run
)

target_link_libraries(ncnn_llm_rag_app PRIVATE
  ${NCNN_TARGET}
  nlohmann_json::nlohmann_json
  httplib::httplib
  Threads::Threads
)

target_compile_definitions(ncnn_llm_rag_app PRIVATE NCNN_LLM_WITH_OPENCV=0)
if (WIN32)
  target_compile_definitions(ncnn_llm_rag_app PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Link OpenMP runtime after ncnn (important when ncnn is a static archive).
if (DEFINED NCNN_OPENMP_LIBRARY AND EXISTS "${NCNN_OPENMP_LIBRARY}")
  if (APPLE AND "${NCNN_OPENMP_LIBRARY}" MATCHES "\\.framework/")
    # Prefer framework-style linking on macOS.
    # NCNN_OPENMP_LIBRARY is expected like: .../openmp.framework/Versions/A/openmp
    get_filename_component(_openmp_bin_dir "${NCNN_OPENMP_LIBRARY}" DIRECTORY)            # .../Versions/A
    get_filename_component(_openmp_versions_dir "${_openmp_bin_dir}" DIRECTORY)          # .../Versions
    get_filename_component(_openmp_framework_dir "${_openmp_versions_dir}" DIRECTORY)    # .../openmp.framework
    get_filename_component(_openmp_frameworks_root "${_openmp_framework_dir}" DIRECTORY) # parent dir
    target_link_options(ncnn_llm_rag_app PRIVATE "-F${_openmp_frameworks_root}")
    target_link_libraries(ncnn_llm_rag_app PRIVATE "-framework" "openmp")
  else()
    target_link_libraries(ncnn_llm_rag_app PRIVATE "${NCNN_OPENMP_LIBRARY}")
  endif()
endif()

# Detect whether the linked ncnn provides Vulkan GPU instance API, to avoid link failures on non-vulkan builds.
# (On Windows, backslashes in paths can break CMake's try_compile unless normalized.)
set(_ncnn_has_vulkan_api 0)
if (NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
  file(TO_CMAKE_PATH "${NCNN_INCLUDE_DIR}" _ncnn_inc_cmake)
  file(TO_CMAKE_PATH "${NCNN_LIBRARY}" _ncnn_lib_cmake)

  file(WRITE ${CMAKE_BINARY_DIR}/cmake_check_ncnn_gpu.cpp [=[
#include <ncnn/gpu.h>
int main(){ return ncnn::get_gpu_count(); }
]=])
  try_compile(_ncnn_gpu_api_links
    ${CMAKE_BINARY_DIR}/cmake_check_ncnn_gpu_build
    ${CMAKE_BINARY_DIR}/cmake_check_ncnn_gpu.cpp
    LINK_LIBRARIES "${_ncnn_lib_cmake}"
    CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=-I${_ncnn_inc_cmake}"
  )
  if (_ncnn_gpu_api_links)
    set(_ncnn_has_vulkan_api 1)
  endif()
endif()

if (_ncnn_has_vulkan_api)
  target_compile_definitions(ncnn_llm_rag_app PRIVATE NCNN_RAG_HAS_VULKAN_API=1)
endif()
