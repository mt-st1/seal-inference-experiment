cmake_minimum_required(VERSION 3.15)

project(seal-experiment VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(SEAL 3.6 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)

function(setup_property TARGETS)
  message("Setup")
  foreach(target IN LISTS TARGETS)
    message("Setting ${target}...")
    set_target_properties("${target}"
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
    )
    target_compile_features("${target}" PUBLIC cxx_std_17)
    target_compile_options("${target}"
      PUBLIC
        -Wall
        $<$<CONFIG:Debug>:-O0 -g3>           # -D CMAKE_BUILD_TYPE=Debug
        $<$<CONFIG:Release>:-O3>             # -D CMAKE_BUILD_TYPE=Release
        $<$<CONFIG:MinSizeRel>:-Os>          # -D CMAKE_BUILD_TYPE=MinSizeRel
        $<$<CONFIG:RelWithDebInfo>:-O3 -g3>  # -D CMAKE_BUILD_TYPE=RelWithDebInfo
    )
    target_link_options("${target}" PUBLIC -Wall)
  endforeach()
endfunction()

add_executable(gen_keys
  src/gen_keys.cpp
  src/utils/constants.cpp
  src/utils/helper.cpp
)
add_executable(example
  src/example.cpp
  src/utils/constants.cpp
  src/utils/helper.cpp
)
add_executable(network_sample
  src/network_sample.cpp
  src/utils/constants.cpp
  src/utils/helper.cpp
)

target_link_libraries(gen_keys
  SEAL::seal
  stdc++fs
)
target_link_libraries(example
  SEAL::seal
)
target_link_libraries(network_sample
  SEAL::seal
  cnn
)

add_subdirectory(src/cnn)

set(TARGETS
  gen_keys
  example
  network_sample
)

setup_property("${TARGETS}")
