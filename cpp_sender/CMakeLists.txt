cmake_minimum_required(VERSION 3.10)
project(MavlinkSender)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
    set(LIBS ws2_32)
else()
    set(LIBS pthread)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mavlink/include
)

add_executable(mavlink_sender
    src/main.cpp
    src/mavlink_udp.cpp
    src/simple_control.cpp
)

target_link_libraries(mavlink_sender PRIVATE ${LIBS})

add_executable(simple_udp_sender
    src/simple_udp_sender.cpp
)

target_link_libraries(simple_udp_sender PRIVATE ${LIBS})

add_executable(periodic_sender
    src/periodic_sender.cpp
    src/mavlink_udp.cpp
)

target_link_libraries(periodic_sender PRIVATE ${LIBS})
