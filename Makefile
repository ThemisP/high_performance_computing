stencil: stencil.c
	mpiicc -std=c99 -O3 -xHOST $^ -o $@
