# Build the simulator (build/snn) and the tests (build/test_snn).
#   make  # GPU architecture detected automatically
#   make ARCH=sm_80  # or set it (A100: sm_80, RTX 4090 / 6000 Ada: sm_89, H100: sm_90)
#   make test  # must print ALL TESTS PASSED

NVCC ?= nvcc
# compute capability of the first GPU, e.g. 8.9 -> sm_89 (sm_89 if nvidia-smi is missing)
ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .' | sed 's/^/sm_/'),sm_89)
FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall

all: build/snn build/test_snn

build/snn: src/main.cu src/*.cuh
	@mkdir -p build
	$(NVCC) $(FLAGS) -o $@ src/main.cu

build/test_snn: tests/test_snn.cu src/*.cuh
	@mkdir -p build
	$(NVCC) $(FLAGS) -o $@ tests/test_snn.cu

test: build/test_snn
	./build/test_snn

clean:
	rm -rf build

.PHONY: all test clean
