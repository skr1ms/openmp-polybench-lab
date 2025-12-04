fp: fp.c
	gcc -std=c99 -fopenmp fp.c -o fp
clean:
	rm -f fp