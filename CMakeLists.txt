cmake_minimum_required(VERSION 3.20)

project(chinese_cheese_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(xiangqi_engine
    engine/src/cycle_adjudicator.cpp
    engine/src/evaluation.cpp
    engine/src/game_history.cpp
    engine/src/halfka.cpp
    engine/src/move_ordering.cpp
    engine/src/position.cpp
    engine/src/search.cpp
    engine/src/transposition_table.cpp
)

target_include_directories(xiangqi_engine
    PUBLIC engine/include
)

target_compile_options(xiangqi_engine PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic;-Wconversion>
)

add_executable(xiangqi_cli engine/src/main.cpp)
target_link_libraries(xiangqi_cli PRIVATE xiangqi_engine)

enable_testing()
add_executable(xiangqi_rules_tests engine/tests/rules_tests.cpp)
target_link_libraries(xiangqi_rules_tests PRIVATE xiangqi_engine)
add_test(NAME xiangqi_rules_tests COMMAND xiangqi_rules_tests)
