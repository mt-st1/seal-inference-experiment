set(module_name cnn)

option(BUILD_SHARED_LIBS "build as a shared library" ON)

include_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB sources *.cpp)

if (BUILD_SHARED_LIBS)
  add_library(${module_name} SHARED ${sources})
else()
  add_library(${module_name} STATIC ${sources})
endif()

target_link_libraries(${module_name} SEAL::seal Eigen3::Eigen)
