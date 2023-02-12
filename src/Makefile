CC=mpicc
CFLAGS= -g -Wall -O3


BIN = ../bin
LIB =  -lm -fopenmp -lpthread -lfftw3 -lmpi
INC = -I../include
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)

ifeq ($(GPU),1)
    CFLAGS += -DGPU

    # CUDA_PATH needs to be adapted depending on where CUDA is installed
    #CUDA_PATH=/usr (if you install it via: sudo apt-get install nvidia-cuda-toolbox, cuda version is v10.1)
    #CUDA_PATH=/usr/local/cuda (if you install it via: sudo apt-get install cuda, latest version of cuda is installed in this way)
    CUDA_PATH = /usr
    NC = nvcc
    cuda_flags = -O3 -ccbin=$(CC) -arch=sm_50 #without arch=sm_XX may lead to compilation errors, XX should be modified according to your hardware
    cuda_inc = -I../include -I$(CUDA_PATH)/include
    cuda_lib = -L$(CUDA_PATH)/lib64  -lcudart -lcufft -lstdc++ #-lcublas 
    LIB += $(cuda_lib)
    
    SRC2 = $(wildcard *.cu)
    OBJ +=  $(SRC2:.cu=.o)
endif



all: clean fdtd

%.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@ $(INC) $(LIB)

ifeq ($(GPU),1)
%.o: %.cu
	$(NC) $(cuda_flags) -c  $^  -o $@ $(cuda_inc) $(cuda_lib) 
endif

fdtd:	$(OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/fdtd $(OBJ) $(LIB)

clean:
	find . -name "*.o"   -exec rm {} \;
	find . -name "*.c%"  -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~"    -exec rm {} \;
	find . -name "\#*"   -exec rm {} \;
	rm -f $(OBJ) $(BIN)/fdtd


