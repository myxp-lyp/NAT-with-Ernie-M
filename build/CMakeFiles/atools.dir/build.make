# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /data/yl7622/anaconda3/envs/Detectgpt/bin/cmake

# The command to remove a file.
RM = /data/yl7622/anaconda3/envs/Detectgpt/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build

# Include any dependencies generated for this target.
include CMakeFiles/atools.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/atools.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/atools.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/atools.dir/flags.make

CMakeFiles/atools.dir/src/alignment_io.cc.o: CMakeFiles/atools.dir/flags.make
CMakeFiles/atools.dir/src/alignment_io.cc.o: ../src/alignment_io.cc
CMakeFiles/atools.dir/src/alignment_io.cc.o: CMakeFiles/atools.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/atools.dir/src/alignment_io.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/atools.dir/src/alignment_io.cc.o -MF CMakeFiles/atools.dir/src/alignment_io.cc.o.d -o CMakeFiles/atools.dir/src/alignment_io.cc.o -c /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/src/alignment_io.cc

CMakeFiles/atools.dir/src/alignment_io.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atools.dir/src/alignment_io.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/src/alignment_io.cc > CMakeFiles/atools.dir/src/alignment_io.cc.i

CMakeFiles/atools.dir/src/alignment_io.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atools.dir/src/alignment_io.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/src/alignment_io.cc -o CMakeFiles/atools.dir/src/alignment_io.cc.s

CMakeFiles/atools.dir/src/atools.cc.o: CMakeFiles/atools.dir/flags.make
CMakeFiles/atools.dir/src/atools.cc.o: ../src/atools.cc
CMakeFiles/atools.dir/src/atools.cc.o: CMakeFiles/atools.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/atools.dir/src/atools.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/atools.dir/src/atools.cc.o -MF CMakeFiles/atools.dir/src/atools.cc.o.d -o CMakeFiles/atools.dir/src/atools.cc.o -c /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/src/atools.cc

CMakeFiles/atools.dir/src/atools.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atools.dir/src/atools.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/src/atools.cc > CMakeFiles/atools.dir/src/atools.cc.i

CMakeFiles/atools.dir/src/atools.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atools.dir/src/atools.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/src/atools.cc -o CMakeFiles/atools.dir/src/atools.cc.s

# Object files for target atools
atools_OBJECTS = \
"CMakeFiles/atools.dir/src/alignment_io.cc.o" \
"CMakeFiles/atools.dir/src/atools.cc.o"

# External object files for target atools
atools_EXTERNAL_OBJECTS =

atools: CMakeFiles/atools.dir/src/alignment_io.cc.o
atools: CMakeFiles/atools.dir/src/atools.cc.o
atools: CMakeFiles/atools.dir/build.make
atools: CMakeFiles/atools.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable atools"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/atools.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/atools.dir/build: atools
.PHONY : CMakeFiles/atools.dir/build

CMakeFiles/atools.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/atools.dir/cmake_clean.cmake
.PHONY : CMakeFiles/atools.dir/clean

CMakeFiles/atools.dir/depend:
	cd /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build /data/yl7622/NAT-with-Ernie-M/data/wmt16/en-ro/fast_align/build/CMakeFiles/atools.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/atools.dir/depend

