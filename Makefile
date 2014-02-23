CC = g++

exec: BackPropagation.h BackPropagation.cpp IOFromFile.cpp IOFromFile.h Lib.h Lib.cpp includes.h main.cpp
	$(CC) BackPropagation.cpp IOFromFile.cpp Lib.cpp main.cpp -o neural
