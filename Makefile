main:
	/lusr/cuda-11.6/bin/nvcc -g -o main.o -c main.cpp
	/lusr/cuda-11.6/bin/nvcc -g -o kernels.o -c kernels.cu
	/lusr/cuda-11.6/bin/nvcc -g -o main main.o kernels.o

.PHONY: clean
clean:
	rm -f main
	rm -f kernels.o
	rm -f main.o

.PHONY: test
test:
	./main -i /u/himanshu/concurrency/final-project/input_sequences-big.txt -n 100 -k 400 -s 21 -p


run:
	make clean 
	make main
	make test