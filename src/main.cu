#include "generator.hu"

int main() {
    // Start a biome generation for the nether on seed 420
    generateBiomes<<<1,1>>>(420);
    cudaDeviceSynchronize();
}