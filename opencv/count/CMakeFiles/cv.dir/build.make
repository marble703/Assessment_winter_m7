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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count

# Include any dependencies generated for this target.
include CMakeFiles/cv.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cv.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cv.dir/flags.make

CMakeFiles/cv.dir/temp.cpp.o: CMakeFiles/cv.dir/flags.make
CMakeFiles/cv.dir/temp.cpp.o: temp.cpp
CMakeFiles/cv.dir/temp.cpp.o: CMakeFiles/cv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cv.dir/temp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cv.dir/temp.cpp.o -MF CMakeFiles/cv.dir/temp.cpp.o.d -o CMakeFiles/cv.dir/temp.cpp.o -c /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count/temp.cpp

CMakeFiles/cv.dir/temp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cv.dir/temp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count/temp.cpp > CMakeFiles/cv.dir/temp.cpp.i

CMakeFiles/cv.dir/temp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cv.dir/temp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count/temp.cpp -o CMakeFiles/cv.dir/temp.cpp.s

# Object files for target cv
cv_OBJECTS = \
"CMakeFiles/cv.dir/temp.cpp.o"

# External object files for target cv
cv_EXTERNAL_OBJECTS =

cv: CMakeFiles/cv.dir/temp.cpp.o
cv: CMakeFiles/cv.dir/build.make
cv: /usr/local/lib/libopencv_gapi.so.4.9.0
cv: /usr/local/lib/libopencv_highgui.so.4.9.0
cv: /usr/local/lib/libopencv_ml.so.4.9.0
cv: /usr/local/lib/libopencv_objdetect.so.4.9.0
cv: /usr/local/lib/libopencv_photo.so.4.9.0
cv: /usr/local/lib/libopencv_stitching.so.4.9.0
cv: /usr/local/lib/libopencv_video.so.4.9.0
cv: /usr/local/lib/libopencv_videoio.so.4.9.0
cv: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
cv: /usr/local/lib/libopencv_dnn.so.4.9.0
cv: /usr/local/lib/libopencv_calib3d.so.4.9.0
cv: /usr/local/lib/libopencv_features2d.so.4.9.0
cv: /usr/local/lib/libopencv_flann.so.4.9.0
cv: /usr/local/lib/libopencv_imgproc.so.4.9.0
cv: /usr/local/lib/libopencv_core.so.4.9.0
cv: CMakeFiles/cv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cv.dir/build: cv
.PHONY : CMakeFiles/cv.dir/build

CMakeFiles/cv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cv.dir/clean

CMakeFiles/cv.dir/depend:
	cd /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count /media/chenyian/Data/programme/Visual/Assessment_winter_m7/opencv/count/CMakeFiles/cv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cv.dir/depend

