# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.12.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.12.4/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/inception_client.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/inception_client.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/inception_client.dir/flags.make

CMakeFiles/inception_client.dir/inception_client.cc.o: CMakeFiles/inception_client.dir/flags.make
CMakeFiles/inception_client.dir/inception_client.cc.o: ../inception_client.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/inception_client.dir/inception_client.cc.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inception_client.dir/inception_client.cc.o -c /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/inception_client.cc

CMakeFiles/inception_client.dir/inception_client.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inception_client.dir/inception_client.cc.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/inception_client.cc > CMakeFiles/inception_client.dir/inception_client.cc.i

CMakeFiles/inception_client.dir/inception_client.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inception_client.dir/inception_client.cc.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/inception_client.cc -o CMakeFiles/inception_client.dir/inception_client.cc.s

# Object files for target inception_client
inception_client_OBJECTS = \
"CMakeFiles/inception_client.dir/inception_client.cc.o"

# External object files for target inception_client
inception_client_EXTERNAL_OBJECTS =

inception_client: CMakeFiles/inception_client.dir/inception_client.cc.o
inception_client: CMakeFiles/inception_client.dir/build.make
inception_client: libtfserving.dylib
inception_client: /usr/local/lib/libgrpc++_reflection.dylib
inception_client: /usr/local/lib/libgrpc++.dylib
inception_client: /usr/local/lib/libgrpc.dylib
inception_client: /usr/local/lib/libprotobuf.dylib
inception_client: CMakeFiles/inception_client.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable inception_client"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/inception_client.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/inception_client.dir/build: inception_client

.PHONY : CMakeFiles/inception_client.dir/build

CMakeFiles/inception_client.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/inception_client.dir/cmake_clean.cmake
.PHONY : CMakeFiles/inception_client.dir/clean

CMakeFiles/inception_client.dir/depend:
	cd /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/build /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/build /Users/liulu51/lu-personal-file/lu-github/Hello-Machine-Learning/famous-frameworks/tensorflow/serving/inception-model/client_cpp/build/CMakeFiles/inception_client.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/inception_client.dir/depend

