CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors -std=c99
LIBS = -lm

# specify the target executable and the source files needed to build it
symnmf: symnmf.o symnmf.h
	$(CC) -o symnmf $(CFLAGS) symnmf.o $(LIBS)
# specify the object files that are generated from the corresponding source files
symnmf.o: symnmf.c
	$(CC) -c $(CFLAGS) symnmf.c $(LIBS)