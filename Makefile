CC = nvcc
SOURCES = $(wildcard src/*.cu)
OBJECTS = $(SOURCES:.cu=.obj)

%.obj: %.cu
	nvcc.exe -dc $< -o $@

build: $(OBJECTS)
	nvcc.exe $^ -o main

run: build
	.\main.exe

clean:
	del /Q /F /S *.obj *.exe *.exp *.lib