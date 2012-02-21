file(GLOB MATCH_LIBRARY_SOURCES ${MATCH_SOURCE_DIR}/src/*.cu)

include_directories(${CUDA_INCLUDE_DIR} ${TBB_INCLUDE_DIR} ${MATCH_SOURCE_DIR}/src/)
cuda_add_library(matcher ${MATCH_LIBRARY_SOURCES})

