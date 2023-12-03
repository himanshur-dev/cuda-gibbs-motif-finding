main:
	/lusr/cuda-11.6/bin/nvcc -o main.o -c main.cpp
	/lusr/cuda-11.6/bin/nvcc -o kernels.o -c kernels.cu
	/lusr/cuda-11.6/bin/nvcc -o main main.o kernels.o

.PHONY: clean
clean:
	rm -f main
	rm -f kernels.o
	rm -f main.o

.PHONY: test
test:
	./main -i /u/himanshu/concurrency/final-project/data/base-data-file.txt -n 1600 -k 10 -p
run:
	make clean 
	make main
	make test