NVCC  ?= nvcc
ARCH  ?= sm_89            # RTX 4090. A100: sm_80, H100: sm_90
FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall

all: build/snn build/test_snn

build/snn: src/main.cu src/snn.cuh
	@mkdir -p build
	$(NVCC) $(FLAGS) -o $@ src/main.cu

build/test_snn: tests/test_snn.cu src/snn.cuh
	@mkdir -p build
	$(NVCC) $(FLAGS) -o $@ tests/test_snn.cu

test: build/test_snn
	./build/test_snn

clean:
	rm -rf build

.PHONY: all test clean
