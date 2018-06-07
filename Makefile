CCOMP ?= icpc
CFLAGS ?= -O3 -g -qopenmp

all:
	$(CCOMP) $(CFLAGS) gups.c -o GUPS.x
	$(CCOMP) $(CFLAGS) gups.c -S
