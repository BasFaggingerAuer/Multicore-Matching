include_directories(${CUDA_INCLUDE_DIR} ${TBB_INCLUDE_DIR} ${MATCH_SOURCE_DIR}/src/)

cuda_add_executable(gpustat gpustat.cu)
target_link_libraries(gpustat ${CUDA_LIBRARY})

find_package(Boost COMPONENTS iostreams program_options REQUIRED)

link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(match match.cpp)
target_link_libraries(match ${CUDA_LIBRARY})
target_link_libraries(match ${TBB_LIBRARY})
target_link_libraries(match ${Boost_LIBRARIES})
target_link_libraries(match matcher)

add_executable(graphstat graphstat.cpp)
target_link_libraries(graphstat ${Boost_LIBRARIES})
