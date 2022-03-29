CC=gcc # mpicc
CFLAGS= -g -Wall 

# FFTW_INC =/home/pyang/software_alias/external/gcc/include/fftw # /applis/site/stow/gcc_4.4.6/fftw_3.3.3/include
# FFTW_LIB =/home/pyang/software_alias/external/gcc/lib64  # /applis/site/stow/gcc_4.4.6/fftw_3.3.3/lib

# NETCDF_INC =/home/pyang/software_alias/external/gcc/include/netcdf # /applis/site/stow/gcc_4.4.6/netcdf_3.3.3/include
# NETCDF_LIB =/home/pyang/software_alias/external/gcc/lib64  # /applis/site/stow/gcc_4.4.6/netcdf_3.3.3/lib


BIN = .
# LIB = -L$(FFTW_LIB) -lfftw3  -L$(NETCDF_LIB) -lnetcdf -lm -lmpi -fopenmp
# INC = -I. -I$(FFTW_INC) -I$(NETCDF_INC)
LIB = -lm 
INC = -I. 
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)



all: clean main

%.o: %.c 
	$(CC) $(CFLAGS) -c $^ -o $@ $(INC) $(LIB)

main:	$(OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/main $(OBJ) $(LIB)

clean:
	find . -name "*.o"   -exec rm {} \;
	find . -name "*.c%"  -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~"    -exec rm {} \;
	find . -name "\#*"   -exec rm {} \;
	rm -f $(OBJ) main


