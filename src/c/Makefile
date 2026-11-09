CC = gcc
CFLAGS = -O3 -ffast-math -march=native -std=c11 -Wall -Wextra
LDFLAGS = -lm
OPENMP = -fopenmp

SRC = graphene_mc.c
BIN = graphene_mc
LIB = libgraphene_mc.so

.PHONY: all clean

all: $(BIN) $(LIB)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(OPENMP) -o $(BIN) $(SRC) $(LDFLAGS)

$(LIB): $(SRC)
	$(CC) $(CFLAGS) $(OPENMP) -fPIC -shared -o $(LIB) $(SRC) $(LDFLAGS)

clean:
	rm -f $(BIN) $(LIB)
