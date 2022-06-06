#include "imagery.hu"
#include <stdio.h>

bool save_ppm(unsigned char* pixels, uint64_t width, uint64_t height) {
    if (pixels) {
        FILE* out = fopen("subscribe.ppm", "wb");
        if (out) {
            fprintf(out, "P6\n%llu %llu\n255\n", width, height);
            unsigned long chunks_written = fwrite(pixels, width * height * 3, 1, out);
            fclose(out);
            return (bool)chunks_written;
        }
    }
    return false;
}

int main() {
    nether_noise* d_noise;
    cudaMalloc(&d_noise, sizeof(nether_noise));

    int64_t length = 1024*32;
    int64_t blockCount = length/16;

    dim3 threads(16, 16);
    dim3 blocks(blockCount, blockCount);

    size_t len = length*length*3;
    unsigned char* h_array = (unsigned char*) malloc(len);
    unsigned char* d_array;
    cudaMalloc(&d_array, len);

    make<<<1,1>>>(d_noise, 420);    
    fill<<<blocks,threads>>>(d_noise, length, d_array);
    printf("%s\n", cudaGetErrorString(cudaPeekAtLastError()));
    cudaDeviceSynchronize();

    cudaMemcpy(h_array, d_array, len, cudaMemcpyDeviceToHost);

    cudaFree(d_array);

    save_ppm(h_array, length, length);
}