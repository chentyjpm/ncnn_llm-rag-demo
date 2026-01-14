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

find_package(ncnn CONFIG REQUIRED)
if (TARGET ncnn::ncnn)
  set(NCNN_TARGET ncnn::ncnn)
elseif (TARGET ncnn)
  set(NCNN_TARGET ncnn)
else()
  message(FATAL_ERROR "ncnn package found but no target (expected ncnn::ncnn or ncnn).")
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

# Ensure cpp-httplib builds in HTTP-only mode (no OpenSSL runtime dependency like libssl-3-x64.dll).
# Some environments may inject CPPHTTPLIB_OPENSSL_SUPPORT via global compiler flags.
if (MSVC)
  target_compile_options(ncnn_llm_rag_app PRIVATE /UCPPHTTPLIB_OPENSSL_SUPPORT)
else()
  target_compile_options(ncnn_llm_rag_app PRIVATE -UCPPHTTPLIB_OPENSSL_SUPPORT)
endif()

# Workaround for some ncnn CMake packages that export include dir as ".../include/ncnn".
# Our code uses "<ncnn/*.h>", which requires adding the parent include dir as well.
get_target_property(_ncnn_iface_includes ${NCNN_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
if (_ncnn_iface_includes)
  set(_ncnn_extra_includes "")
  foreach(_inc IN LISTS _ncnn_iface_includes)
    if (_inc MATCHES "(/|\\\\)include(/|\\\\)ncnn$")
      get_filename_component(_parent "${_inc}" DIRECTORY)
      list(APPEND _ncnn_extra_includes "${_parent}")
    endif()
  endforeach()
  if (_ncnn_extra_includes)
    list(REMOVE_DUPLICATES _ncnn_extra_includes)
    target_include_directories(ncnn_llm_rag_app PRIVATE ${_ncnn_extra_includes})
  endif()
endif()

# Gate optional Vulkan GPU instance calls based on whether headers expose the API.
set(_ncnn_has_gpu_instance_api 0)
set(_ncnn_check_includes "")
if (_ncnn_iface_includes)
  list(APPEND _ncnn_check_includes ${_ncnn_iface_includes})
endif()
if (DEFINED _ncnn_extra_includes AND _ncnn_extra_includes)
  list(APPEND _ncnn_check_includes ${_ncnn_extra_includes})
endif()
if (_ncnn_check_includes)
  set(_old_req_inc "${CMAKE_REQUIRED_INCLUDES}")
  set(CMAKE_REQUIRED_INCLUDES ${_ncnn_check_includes})
  check_cxx_source_compiles([=[
#include <ncnn/gpu.h>
int main() {
  ncnn::create_gpu_instance();
  int c = ncnn::get_gpu_count();
  ncnn::destroy_gpu_instance();
  return c;
}
]=] _ncnn_gpu_instance_compiles)
  set(CMAKE_REQUIRED_INCLUDES "${_old_req_inc}")
  if (_ncnn_gpu_instance_compiles)
    set(_ncnn_has_gpu_instance_api 1)
  endif()
endif()

if (_ncnn_has_gpu_instance_api)
  target_compile_definitions(ncnn_llm_rag_app PRIVATE NCNN_RAG_HAS_VULKAN_API=1)
endif()
