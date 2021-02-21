CC = gcc 

LIBS=-lm
OBJ=svdlib.o svdutil.o las2.o rand.o matops.o OptSpace.o 

libOptSpace.a: ${PWD} ${OBJ}
	rm -f ${PWD}/$@
	ar cr $@ ${OBJ}
	ranlib $@
svdlib.o: Makefile svdlib.h svdlib.c
	${CC} ${CFLAGS} -c svdlib.c
svdutil.o: Makefile svdutil.c svdutil.h
	${CC} ${CFLAGS} -c svdutil.c
las2.o: Makefile las2.c svdlib.h svdutil.h
	${CC} ${CFLAGS} -c las2.c
rand.o: Makefile rand.c 
	 ${CC} ${CFLAGS} -c rand.c
OptSpace.o: Makefile OptSpace.c 
	 ${CC} ${CFLAGS} -c OptSpace.c
matops.o: Makefile matops.c 
	 ${CC} ${CFLAGS} -c matops.c
clean: 
	rm *.o

