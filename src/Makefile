CC=mpicc
CFLAGS= -g -Wall -O3


BIN = ../bin
LIB =  -lm -fopenmp -lpthread -lfftw3 -lmpi
INC = -I../include
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)

ifeq ($(GPU),1)
    CFLAGS += -DGPU
    
    #assume CUDA is installed in the path $(CUDA_PATH)
    NC = nvcc
    cuda_flags = -O3 -ccbin=$(CC) -arch=sm_50 #without arch=sm_XX may lead to compilation errors, XX should be modified according to your hardware
    cuda_inc = -I../include -I$(CUDA_PATH)/include
    cuda_lib = -L$(CUDA_PATH)/lib64  -lcudart -lcublas -lcufft -lstdc++
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


