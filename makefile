COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-9.2/samples/common/inc 
EXES = example
all: ${EXES}


example:   example.cu
	${COMPILER} ${CFLAGS} example.cu  -o example

%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} $< -c 

clean:
	rm -f *.o *~ ${EXES} ${CFILES}
