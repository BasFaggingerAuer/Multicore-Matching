file(GLOB MATCH_LIBRARY_SOURCES ${MATCH_SOURCE_DIR}/src/*.cu)
file (GLOB DOT_LIBRARY_SOURCES ${MATCH_SOURCE_DIR}/DotWriter/lib/*.cpp)
file (GLOB DOT_LIBRARY_HEADERS ${MATCH_SOURCE_DIR}/DotWriter/lib/*.h)

include_directories(${CUDA_INCLUDE_DIR} ${TBB_INCLUDE_DIR} ${MATCH_SOURCE_DIR}/src/ ${MATCH_SOURCE_DIR}/DotWriter/lib)
cuda_add_library(matcher ${MATCH_LIBRARY_SOURCES})

