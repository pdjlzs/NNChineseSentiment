project(MinervaSegs)

include_directories(
  basic
  /opt/mshadow/
  ../LibN3L/
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILdtype MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda

#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)

add_executable(LSTMWordDetector LSTMWordDetector.cpp)
add_executable(LSTMWordDetectorJoint LSTMWordDetectorJoint.cpp)
add_executable(SparseDetectorBi SparseDetectorBi.cpp)
add_executable(SparseDetectorBiPrfx SparseDetectorBiPrfx.cpp)
add_executable(SparseDetectorBiPUniJoint SparseDetectorBiPUniJoint.cpp)
add_executable(SparseDetectorBiPBiJoint SparseDetectorBiPBiJoint.cpp)
add_executable(SparseDetectorTri SparseDetectorTri.cpp)
add_executable(SparseDetectorTriPrfx SparseDetectorTriPrfx.cpp)
add_executable(SparseDetectorTriPUniJoint SparseDetectorTriPUniJoint.cpp)
add_executable(SparseDetectorTriPBiJoint SparseDetectorTriPBiJoint.cpp)
add_executable(VSparseDetectorBi VSparseDetectorBi.cpp)
add_executable(VSparseDetectorBiPrfx VSparseDetectorBiPrfx.cpp)
add_executable(VSparseDetectorTri VSparseDetectorTri.cpp)
add_executable(VSparseDetectorTriPrfx VSparseDetectorTriPrfx.cpp)

target_link_libraries(LSTMWordDetector openblas)
target_link_libraries(LSTMWordDetectorJoint openblas)
target_link_libraries(SparseDetectorBi openblas)
target_link_libraries(SparseDetectorBiPrfx openblas)
target_link_libraries(SparseDetectorBiPUniJoint openblas)
target_link_libraries(SparseDetectorBiPBiJoint openblas)
target_link_libraries(SparseDetectorTri openblas)
target_link_libraries(SparseDetectorTriPrfx openblas)
target_link_libraries(SparseDetectorTriPUniJoint openblas)
target_link_libraries(SparseDetectorTriPBiJoint openblas)
target_link_libraries(VSparseDetectorBi openblas)
target_link_libraries(VSparseDetectorBiPrfx openblas)
target_link_libraries(VSparseDetectorTri openblas)
target_link_libraries(VSparseDetectorTriPrfx openblas)
