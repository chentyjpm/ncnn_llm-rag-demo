cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED exe OR NOT DEFINED dest)
  message(FATAL_ERROR "Usage: -Dexe=... -Ddest=...")
endif()

file(MAKE_DIRECTORY "${dest}")

# On Windows, this uses dumpbin/llvm-objdump depending on the toolchain.
set(resolved_deps "")
set(unresolved_deps "")

file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${exe}"
  RESOLVED_DEPENDENCIES_VAR resolved_deps
  UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
)

foreach(d IN LISTS unresolved_deps)
  message(STATUS "Unresolved runtime dependency: ${d}")
endforeach()

foreach(d IN LISTS resolved_deps)
  # Avoid copying system DLLs.
  if(d MATCHES "^[A-Za-z]:[/\\\\]Windows[/\\\\]" OR d MATCHES "^[A-Za-z]:[/\\\\]WINDOWS[/\\\\]")
    continue()
  endif()
  get_filename_component(name "${d}" NAME)
  file(COPY_FILE "${d}" "${dest}/${name}" ONLY_IF_DIFFERENT)
endforeach()

