# Compiler and Flags
NVCC = nvcc
CXXFLAGS = -Xcompiler -fPIC
PYTHON = python3

# Output file
TARGET = ryupy.so
SRC_DIR = src
CUDA_STATIC_LIB_DIR = /nix/store/yp5wra915j9p5nxa631svxv0x1r5z3m3-cuda_cudart-11.8.89-static/lib

# Source and object files
SRCS := $(shell find $(SRC_DIR) -type f \( -name "*.cu" -o -name "*.cpp" \))
OBJS := $(SRCS:.cu=.o)
OBJS := $(OBJS:.cpp=.o)

# Python include and library directories
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_DIR := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_LIB := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))")

# Pybind11 includes
PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

# Build the shared library
$(TARGET): $(OBJS)
	$(NVCC) -shared -arch=sm_75 -DCUBLAS_ENABLED -lcublas -lcudnn -lcurand -lcudadevrt -lcudart_static \
		-L$(CUDA_STATIC_LIB_DIR) $(OBJS) $(PYBIND_INCLUDES) -I$(PYTHON_INCLUDE) -L$(PYTHON_LIB_DIR) -lpython3.10 \
		-o $(TARGET) -std=c++14
	@echo "Successfully built $(TARGET)"

# Compile .cu files to .o
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -I$(PYTHON_INCLUDE) -c $< -o $@

# Compile .cpp files to .o
%.o: %.cpp
	$(NVCC) $(CXXFLAGS) -I$(PYTHON_INCLUDE) -c $< -o $@

