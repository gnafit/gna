find_package(fmt QUIET)
if(NOT fmt_FOUND)
    message(STATUS "libfmt not found, downloading from git")
    include(FetchContent)
    # macro taken from https://cliutils.gitlab.io/modern-cmake/chapters/projects/fetch.html
    # used to handle missing functionality in CMake 3.11-3.14
    if(${CMAKE_VERSION} VERSION_LESS 3.14)
        macro(FetchContent_MakeAvailable NAME)
            FetchContent_GetProperties(${NAME})
            if(NOT ${NAME}_POPULATED)
                FetchContent_Populate(${NAME})
                add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR})
            endif()
        endmacro()
    endif()

    FetchContent_Declare(
      fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG        8.0.1
    )
    FetchContent_MakeAvailable(fmt)
endif()
