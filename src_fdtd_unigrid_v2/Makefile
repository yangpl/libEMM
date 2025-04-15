CC=mpicc # Intel compiler
CFLAGS= -g -Wall 


BIN = .
LIB =  -lm -lmpi -fopenmp -lpthread -lfftw3
INC = -I.
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)


all: clean fdtd

%.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@ $(INC) $(LIB)

fdtd:	$(OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/fdtd $(OBJ) $(LIB)

clean:
	find . -name "*.o"   -exec rm {} \;
	find . -name "*.c%"  -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~"    -exec rm {} \;
	find . -name "\#*"   -exec rm {} \;
	rm -f $(OBJ) fdtd *.bin snapshot* stf


