CC = gcc 

LIBS=-lm

test: Makefile testing.o
	${CC} ${CFLAGS} -o test testing.o ../libOptSpace.a ${LIBS}

testing.o: Makefile testing.c svdlib.h OptSpace.h
	${CC} ${CFLAGS} -c testing.c

clean: 
	rm *.o

