stencil: stencil.c
	icc -std=c99 -O3 -xHOST $^ -o $@
