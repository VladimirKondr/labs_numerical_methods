add_library(linalg
    matrix.cpp
    vector.cpp
    operations.cpp
)

target_compile_options(linalg PRIVATE
    -Wall -Wextra -O2
)