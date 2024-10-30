
cmake_minimum_required(VERSION 3.10)
project(MIOpenConvolutionExample)

# Set HIP standard
set(CMAKE_HIP_STANDARD 14)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Find HIP
find_package(hip REQUIRED CONFIG)
if (NOT hip_FOUND)
	message(FATAL_ERROR "HIP package not found.")
else()
	message(STATUS "HIP found")
endif()


# Find MIOpen
find_package(miopen REQUIRED CONFIG)
if (NOT miopen_FOUND)
	message(FATAL_ERROR "MIOpen package not found.")
else()
	message(STATUS "MIOpen found")
endif()

# Debugging: Print variables provided by HIP
message(STATUS "hip_INCLUDE_DIRS: ${hip_INCLUDE_DIRS}")
message(STATUS "hip_LIBRARIES: ${hip_LIBRARIES}")
message(STATUS "hip_LIB_INSTALL_DIR: ${hip_LIB_INSTALL_DIR}")





# Get the root directory of MIOpen installation
get_filename_component(MIOPEN_ROOT "${miopen_DIR}/../../.." ABSOLUTE)

# Set the path to MIOpen.dll
set(MIOPEN_DLL_PATH "${MIOPEN_ROOT}/bin/MIOpen.dll")

# Verify that MIOpen.dll exists
if (NOT EXISTS "${MIOPEN_DLL_PATH}")
	message(FATAL_ERROR "MIOpen.dll not found at expected path: ${MIOPEN_DLL_PATH}")
endif()




# Add executable
add_executable(convolution_example demo.cpp)



# Include directories
target_include_directories(convolution_example PRIVATE ${hip_INCLUDE_DIRS} ${MIOPEN_INCLUDE_DIRS})

# Link directories
target_link_directories(convolution_example PRIVATE "${hip_LIB_INSTALL_DIR}")

# Link libraries
target_link_libraries(convolution_example PRIVATE ${MIOPEN_LIBRARIES} amdhip64.lib)


# Copy MIOpen.dll to the output directory after building
message(STATUS "copy the MIOpen.dll inside the project.")
add_custom_command(TARGET convolution_example POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${MIOPEN_DLL_PATH}"
	$<TARGET_FILE_DIR:convolution_example>)



# for Visual Studio, set 'convolution_example' as startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT convolution_example)

