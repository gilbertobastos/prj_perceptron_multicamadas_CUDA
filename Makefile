CODE=sm_61
ARCH=compute_61
CXX=g++
CXXFLAGS=-O3 -march=native
CC=gcc
CFLAGS=-O3
NVCC=nvcc
NVCCFLAGS=-O3 -code=$(CODE) -arch=$(ARCH) -Xptxas -O3

prj_perceptron_multicamadas: main.o host_perceptron_multicamadas.o \
perceptron_multicamadas_kernels.o uniform.o 
	$(NVCC) -link main.o host_perceptron_multicamadas.o \
	perceptron_multicamadas_kernels.o uniform.o \
	-o prj_perceptron_multicamadas

main.o: src/main.cu
	$(NVCC) --device-c src/main.cu -lcuda -lcudart \
	-I /usr/local/cuda/include -o main.o

host_perceptron_multicamadas.o: src/host_perceptron_multicamadas.cu
	$(NVCC) --device-c src/host_perceptron_multicamadas.cu $(NVCCFLAGS) \
	-o host_perceptron_multicamadas.o

perceptron_multicamadas_kernels.o: src/perceptron_multicamadas_kernels.cu
	$(NVCC) --device-c src/perceptron_multicamadas_kernels.cu $(NVCCFLAGS) \
	-o perceptron_multicamadas_kernels.o

uniform.o: src/uniform.c
	$(CXX) -c src/uniform.c $(CXXFLAGS) \
	-o uniform.o

clean:
	rm *.o
