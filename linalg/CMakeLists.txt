add_library(linalg
    matrix.cpp
    vector.cpp
    operations.cpp
    solvers.cpp
)

target_compile_options(linalg PRIVATE
    -Wall -Wextra -O2
)

find_package(Boost REQUIRED CONFIG)

target_include_directories(linalg PRIVATE ${Boost_INCLUDE_DIRS})