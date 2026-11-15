NVCC ?= nvcc
# tensor cores: Turing (7.5); add -gencode lines for newer GPUs
GENCODE ?= -gencode arch=compute_75,code=sm_75
NVFLAGS ?= -O3 -std=c++17 -lineinfo $(GENCODE) -Xcompiler -Wall
HEADERS := src/flash.cuh src/standard.cuh src/common.cuh

all: build/attn_test build/attn_bench

build/attn_test: src/test.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) $< -o $@ -lcublas

build/attn_bench: src/bench.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) $< -o $@ -lcublas

build/ptxas.txt: src/bench.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) -Xptxas -v -c $< -o build/ptxas.o 2> $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
