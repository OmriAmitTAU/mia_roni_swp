CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LIBS = -lm


# Specify the target executable and the source files needed to build it
symnmf: symnmf.o symnmf.h
	$(CC) -o symnmf $(CFLAGS) symnmf.o $(LIBS)
# Specify the object files that are generated from the corresponding source files
symnmf.o: symnmf.c
	$(CC) -c $(CFLAGS) symnmf.c $(LIBS)