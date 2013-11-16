all:
	g++ -O3 -I /usr/local/include/ -L /usr/local/lib -o lbmgl.x -framework OpenGL -framework Cocoa -framework IOKit lbmgl.cpp -lglfw
	nvcc -I /usr/local/include/ -L /usr/local/lib/ -arch=sm_30 gpu_lbm.cu -o gpu_lbm.x -Xlinker -framework,OpenGL,-framework,Cocoa,-framework,IOKit -lglfw

clean:
	rm *.x
