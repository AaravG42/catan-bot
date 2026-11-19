CC ?= cc
CFLAGS ?= -O3 -march=native -fPIC -std=c11 -Wall -Wextra -Wno-unused-parameter
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  LIB = libcatan.dylib
  LDFLAGS = -dynamiclib
  # Apple clang does not accept -march=native on arm64 before clang 15; -mcpu=native works.
  CFLAGS := $(subst -march=native,-mcpu=native,$(CFLAGS))
else
  LIB = libcatan.so
  LDFLAGS = -shared
endif

all: $(LIB) bench

$(LIB): catan.c vec.c search.c feats.c catan.h topology.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ catan.c vec.c search.c feats.c -lm

bench: bench.c catan.c vec.c catan.h topology.h
	$(CC) $(CFLAGS) -o $@ bench.c catan.c vec.c

topology.h: gen_topology.py
	PYTHONHASHSEED=0 ../.venv/bin/python gen_topology.py

clean:
	rm -f libcatan.so libcatan.dylib bench

.PHONY: all clean
