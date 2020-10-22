# Compiler
CXX = g++
CXXFLAGS = -std=c++17

# Directory where objects and executables must be saved
BUILD_DIR = build

# Include directories for external libraries (e.g. boost, eigen3, etc.)
EXTRA_INCLUDES = -I /opt/local/include/

# Headers + external libraries
INCLUDES = -I src/ $(EXTRA_INCLUDES)

# Libraries to link
LIB_DIR_FLAGS = -L/opt/local/lib/
LIB_FLAGS$ = -lboost_program_options-mt -lboost_filesystem-mt -lfmt
LIBS = $(LIB_DIR_FLAGS) $(LIB_FLAGS)

# Source and object files necessary for data conversion
DATA_CONVERSION_SOURCES = src/convert_messages.cpp\
					      src/converter/MessageConverter.cpp\
					      src/converter/TA3MessageConverter.cpp\
					      src/utils/FileHandler.cpp\
					      src/utils/EigenExtensions.cpp\
						  src/utils/Tensor3.cpp

DATA_CONVERSION_OBJECTS = $(DATA_CONVERSION_SOURCES:.cpp=.o)

# Phony targets
.PHONY: all
.PHONY: build
.PHONY: sync
#.PHONY: clean

all: build sync

convert_messages: build $(DATA_CONVERSION_OBJECTS)
	$(CXX) $(LIBS) $(patsubst %.o, $(BUILD_DIR)/%.o , $(DATA_CONVERSION_OBJECTS)) -o $(BUILD_DIR)/bin/$@

%.o: %.cpp
	@mkdir -p $(BUILD_DIR)/$(dir $@)
	$(CXX) $(CXXFLAGS) -c $< $(INCLUDES) -o $(BUILD_DIR)/$@

sync:
	./tools/sync_asist_data

build:
	@mkdir -p $(BUILD_DIR)/bin