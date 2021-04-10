TARGET = program 

PROC = $(shell uname -m)
ARCH = $(PROC)-linux

ifeq ($(PROC), x86_64)
CFLAGS := -DUSE_NVJPEG=0 -O3 -fPIC
LIBRARIES := -lnvjpeg -ljpeg
else ifeq ($(PROC), aarch64)
CFLAGS := -DJETSON=1 -march=armv8-a -O3 -fPIC
LIBRARIES := -ljpeg
endif

GL_INC_DIR = /usr/include/GL
GL_LIB_DIR = /usr/lib/$(ARCH)-gnu
CUDA_INC_DIR = /usr/local/cuda-11.0/include
CUDA_LIB_DIR = /usr/local/cuda-11.0/lib64

OBJ_DIR = obj
BIN_PATH = ./$(TARGET)

PKGCFG	= pkg-config
MKDIR	= mkdir
RM	= rm -f
RMDIR	= rm -rf
CXX	= g++
NVCC	= nvcc -ccbin $(CXX)
MAKE	= make
CP	= cp

IFLAGS = -I$(CUDA_INC_DIR) -I$(GL_INC_DIR) -I. -I../..

WFLAGS = -Wall -Wextra -Werror=float-equal -Wuninitialized -Wunused-variable #-Wdouble-promotion
CFLAGS += $(WFLAGS)
NVCCFLAGS = -m64 $(addprefix -Xcompiler ,$(CFLAGS)) $(IFLAGS)

LDFLAGS = -rpath='$$ORIGIN'
LIBRARIES += -L$(BIN_DIR) -L$(CUDA_LIB_DIR) -L$(GL_LIB_DIR) -lGL -lX11 -lEGL -lGLU -lpthread -lz -lcudart -lcudnn -lcublas -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lnvrtc -llz4

NVLDFLAGS = -m64 $(addprefix -Xcompiler ,$(CFLAGS)) $(addprefix -Xlinker ,$(LDFLAGS)) $(LIBRARIES)
GENCODE_FLAGS =
SMS = 53 61 70 72 75
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM := $(lastword $(sort $(SMS)))
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

SOURCES  := $(wildcard ./*.cpp) $(wildcard ../../*.cpp) $(wildcard ./*.cu) $(wildcard ../../*.cu)
INCLUDES := $(wildcard ./*.h ) $(wildcard ../../*.h )
OBJECTS  := $(patsubst ./%.cpp, $(OBJ_DIR)/%.o, $(patsubst ../../%.cpp, $(OBJ_DIR)/%.o, $(patsubst ./%.cu, $(OBJ_DIR)/%.o, $(patsubst ../../%.cu, $(OBJ_DIR)/%.o, $(SOURCES)))))


all: $(BIN_PATH)

$(OBJ_DIR)/%.o : ./%.cpp $(INCLUDES) $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@
$(OBJ_DIR)/%.o : ../../%.cpp $(INCLUDES) $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

$(OBJ_DIR)/%.o: ./%.cu $(INCLUDES) $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@
$(OBJ_DIR)/%.o: ../../%.cu $(INCLUDES) $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

$(BIN_PATH): $(OBJECTS) $(BIN_DIR)
	$(NVCC) $(NVLDFLAGS) -o $@ $(OBJECTS) $(GENCODE_FLAGS)

clean:
	$(RMDIR) $(OBJ_DIR)

$(OBJ_DIR):
	$(MKDIR) -p $(OBJ_DIR)

$(BIN_DIR):
	$(MKDIR) -p $(BIN_DIR)

