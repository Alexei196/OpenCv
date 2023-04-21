# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/andrew/Documents/cvTest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andrew/Documents/cvTest

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/andrew/Documents/cvTest/CMakeFiles /home/andrew/Documents/cvTest//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/andrew/Documents/cvTest/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named kmeans

# Build rule for target.
kmeans: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 kmeans
.PHONY : kmeans

# fast build rule for target.
kmeans/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/kmeans.dir/build.make CMakeFiles/kmeans.dir/build
.PHONY : kmeans/fast

#=============================================================================
# Target rules for targets named sobel

# Build rule for target.
sobel: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 sobel
.PHONY : sobel

# fast build rule for target.
sobel/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sobel.dir/build.make CMakeFiles/sobel.dir/build
.PHONY : sobel/fast

kmeans.o: kmeans.cpp.o
.PHONY : kmeans.o

# target to build an object file
kmeans.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/kmeans.dir/build.make CMakeFiles/kmeans.dir/kmeans.cpp.o
.PHONY : kmeans.cpp.o

kmeans.i: kmeans.cpp.i
.PHONY : kmeans.i

# target to preprocess a source file
kmeans.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/kmeans.dir/build.make CMakeFiles/kmeans.dir/kmeans.cpp.i
.PHONY : kmeans.cpp.i

kmeans.s: kmeans.cpp.s
.PHONY : kmeans.s

# target to generate assembly for a file
kmeans.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/kmeans.dir/build.make CMakeFiles/kmeans.dir/kmeans.cpp.s
.PHONY : kmeans.cpp.s

sobel.o: sobel.cpp.o
.PHONY : sobel.o

# target to build an object file
sobel.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sobel.dir/build.make CMakeFiles/sobel.dir/sobel.cpp.o
.PHONY : sobel.cpp.o

sobel.i: sobel.cpp.i
.PHONY : sobel.i

# target to preprocess a source file
sobel.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sobel.dir/build.make CMakeFiles/sobel.dir/sobel.cpp.i
.PHONY : sobel.cpp.i

sobel.s: sobel.cpp.s
.PHONY : sobel.s

# target to generate assembly for a file
sobel.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sobel.dir/build.make CMakeFiles/sobel.dir/sobel.cpp.s
.PHONY : sobel.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... kmeans"
	@echo "... sobel"
	@echo "... kmeans.o"
	@echo "... kmeans.i"
	@echo "... kmeans.s"
	@echo "... sobel.o"
	@echo "... sobel.i"
	@echo "... sobel.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

