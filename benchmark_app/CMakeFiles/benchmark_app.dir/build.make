# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /opt/intel/openvino/deployment_tools/inference_engine/samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/build

# Include any dependencies generated for this target.
include benchmark_app/CMakeFiles/benchmark_app.dir/depend.make

# Include the progress variables for this target.
include benchmark_app/CMakeFiles/benchmark_app.dir/progress.make

# Include the compile flags for this target's objects.
include benchmark_app/CMakeFiles/benchmark_app.dir/flags.make

benchmark_app/CMakeFiles/benchmark_app.dir/inputs_filling.cpp.o: benchmark_app/CMakeFiles/benchmark_app.dir/flags.make
benchmark_app/CMakeFiles/benchmark_app.dir/inputs_filling.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/inputs_filling.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmark_app/CMakeFiles/benchmark_app.dir/inputs_filling.cpp.o"
	cd /home/pi/build/benchmark_app && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmark_app.dir/inputs_filling.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/inputs_filling.cpp

benchmark_app/CMakeFiles/benchmark_app.dir/inputs_filling.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_app.dir/inputs_filling.cpp.i"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/inputs_filling.cpp > CMakeFiles/benchmark_app.dir/inputs_filling.cpp.i

benchmark_app/CMakeFiles/benchmark_app.dir/inputs_filling.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_app.dir/inputs_filling.cpp.s"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/inputs_filling.cpp -o CMakeFiles/benchmark_app.dir/inputs_filling.cpp.s

benchmark_app/CMakeFiles/benchmark_app.dir/main.cpp.o: benchmark_app/CMakeFiles/benchmark_app.dir/flags.make
benchmark_app/CMakeFiles/benchmark_app.dir/main.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object benchmark_app/CMakeFiles/benchmark_app.dir/main.cpp.o"
	cd /home/pi/build/benchmark_app && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmark_app.dir/main.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/main.cpp

benchmark_app/CMakeFiles/benchmark_app.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_app.dir/main.cpp.i"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/main.cpp > CMakeFiles/benchmark_app.dir/main.cpp.i

benchmark_app/CMakeFiles/benchmark_app.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_app.dir/main.cpp.s"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/main.cpp -o CMakeFiles/benchmark_app.dir/main.cpp.s

benchmark_app/CMakeFiles/benchmark_app.dir/statistics_report.cpp.o: benchmark_app/CMakeFiles/benchmark_app.dir/flags.make
benchmark_app/CMakeFiles/benchmark_app.dir/statistics_report.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/statistics_report.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object benchmark_app/CMakeFiles/benchmark_app.dir/statistics_report.cpp.o"
	cd /home/pi/build/benchmark_app && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmark_app.dir/statistics_report.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/statistics_report.cpp

benchmark_app/CMakeFiles/benchmark_app.dir/statistics_report.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_app.dir/statistics_report.cpp.i"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/statistics_report.cpp > CMakeFiles/benchmark_app.dir/statistics_report.cpp.i

benchmark_app/CMakeFiles/benchmark_app.dir/statistics_report.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_app.dir/statistics_report.cpp.s"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/statistics_report.cpp -o CMakeFiles/benchmark_app.dir/statistics_report.cpp.s

benchmark_app/CMakeFiles/benchmark_app.dir/utils.cpp.o: benchmark_app/CMakeFiles/benchmark_app.dir/flags.make
benchmark_app/CMakeFiles/benchmark_app.dir/utils.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object benchmark_app/CMakeFiles/benchmark_app.dir/utils.cpp.o"
	cd /home/pi/build/benchmark_app && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmark_app.dir/utils.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/utils.cpp

benchmark_app/CMakeFiles/benchmark_app.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_app.dir/utils.cpp.i"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/utils.cpp > CMakeFiles/benchmark_app.dir/utils.cpp.i

benchmark_app/CMakeFiles/benchmark_app.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_app.dir/utils.cpp.s"
	cd /home/pi/build/benchmark_app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app/utils.cpp -o CMakeFiles/benchmark_app.dir/utils.cpp.s

# Object files for target benchmark_app
benchmark_app_OBJECTS = \
"CMakeFiles/benchmark_app.dir/inputs_filling.cpp.o" \
"CMakeFiles/benchmark_app.dir/main.cpp.o" \
"CMakeFiles/benchmark_app.dir/statistics_report.cpp.o" \
"CMakeFiles/benchmark_app.dir/utils.cpp.o"

# External object files for target benchmark_app
benchmark_app_EXTERNAL_OBJECTS =

armv7l/Release/benchmark_app: benchmark_app/CMakeFiles/benchmark_app.dir/inputs_filling.cpp.o
armv7l/Release/benchmark_app: benchmark_app/CMakeFiles/benchmark_app.dir/main.cpp.o
armv7l/Release/benchmark_app: benchmark_app/CMakeFiles/benchmark_app.dir/statistics_report.cpp.o
armv7l/Release/benchmark_app: benchmark_app/CMakeFiles/benchmark_app.dir/utils.cpp.o
armv7l/Release/benchmark_app: benchmark_app/CMakeFiles/benchmark_app.dir/build.make
armv7l/Release/benchmark_app: /opt/intel/openvino/opencv/lib/libopencv_imgcodecs.so.4.1.2
armv7l/Release/benchmark_app: armv7l/Release/lib/libformat_reader.so
armv7l/Release/benchmark_app: armv7l/Release/lib/libcpu_extension.so
armv7l/Release/benchmark_app: armv7l/Release/lib/libgflags_nothreads.a
armv7l/Release/benchmark_app: /opt/intel/openvino/opencv/lib/libopencv_imgproc.so.4.1.2
armv7l/Release/benchmark_app: /opt/intel/openvino/opencv/lib/libopencv_core.so.4.1.2
armv7l/Release/benchmark_app: /opt/intel/openvino/deployment_tools/inference_engine/lib/armv7l/libinference_engine.so
armv7l/Release/benchmark_app: benchmark_app/CMakeFiles/benchmark_app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ../armv7l/Release/benchmark_app"
	cd /home/pi/build/benchmark_app && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmark_app/CMakeFiles/benchmark_app.dir/build: armv7l/Release/benchmark_app

.PHONY : benchmark_app/CMakeFiles/benchmark_app.dir/build

benchmark_app/CMakeFiles/benchmark_app.dir/clean:
	cd /home/pi/build/benchmark_app && $(CMAKE_COMMAND) -P CMakeFiles/benchmark_app.dir/cmake_clean.cmake
.PHONY : benchmark_app/CMakeFiles/benchmark_app.dir/clean

benchmark_app/CMakeFiles/benchmark_app.dir/depend:
	cd /home/pi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /opt/intel/openvino/deployment_tools/inference_engine/samples /opt/intel/openvino/deployment_tools/inference_engine/samples/benchmark_app /home/pi/build /home/pi/build/benchmark_app /home/pi/build/benchmark_app/CMakeFiles/benchmark_app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmark_app/CMakeFiles/benchmark_app.dir/depend

