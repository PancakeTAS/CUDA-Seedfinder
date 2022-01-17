CC = nvcc
SOURCES = $(wildcard src/*.cu)
OBJECTS = $(SOURCES:.cu=.obj)

%.obj: %.cu
	nvcc.exe -dc $< -o $@ -O3 

build: $(OBJECTS)
	nvcc.exe $^ -o main -O3

run: build
	.\main.exe

clean:
	del /Q /F /S *.obj *.exe *.exp *.lib