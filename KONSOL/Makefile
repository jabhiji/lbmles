all:
	g++ -O3 cpu_lbm.cpp -o cpu_lbm.x
	nvcc -arch=sm_30 gpu_lbm.cu -o gpu_lbm.x

clean:
	rm *.x
