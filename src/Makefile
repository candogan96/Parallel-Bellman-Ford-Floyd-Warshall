EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE:=$(EXE_OMP) gen-chain
CFLAGS+=-std=c99 -Wall -Wpedantic -fopenmp -O2

ALL: $(EXE)

omp-bellman-ford: LDLIBS+=-lm

.PHONY: clean

clean:
	\rm -f $(EXE) *.o *~
