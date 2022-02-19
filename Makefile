ifeq ($(OS),Windows_NT)
CC = nvcc.exe
OUT = main.exe
DEL = del /Q /F /S *.obj $(OUT) *.exp *.lib
else
CC = nvcc
OUT = main
DEL = find . -regextype posix-egrep -regex ".*\.(obj|exp|lib)$$" -type f -delete; rm main
endif

SOURCES = $(wildcard src/*.cu)
OBJECTS = $(SOURCES:.cu=.obj)

%.obj: %.cu
	$(CC) -dc $< -o $@

build: $(OBJECTS)
	$(CC) $^ -o $(OUT)

run: build
	./$(OUT)

clean:
	$(DEL)