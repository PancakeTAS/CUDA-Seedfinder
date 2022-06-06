#include "imagery.hu"

__global__ void fill(nether_noise* noise, int64_t length, unsigned char* out) {
    int64_t x = blockIdx.x*16 + threadIdx.x;
    int64_t z = blockIdx.y*16 + threadIdx.y;
    int64_t i = (z*length + x)*3;
    int d = get_nether_biome(noise, x, z);
    switch (get_nether_biome(noise, x, z)) {
        case 0:
            out[i] = 87;
            out[i+1] = 37;
            out[i+2] = 38;
            break;
        case 1:
            out[i] = 77;
            out[i+1] = 58;
            out[i+2] = 46;
            break;
        case 2:
            out[i] = 152;
            out[i+1] = 26;
            out[i+2] = 17;
            break;
        case 3:
            out[i] = 73;
            out[i+1] = 144;
            out[i+2] = 123;
            break;
        case 4:
            out[i] = 100;
            out[i+1] = 95;
            out[i+2] = 99;
            break;
        default:

            break;
    }
}

__global__ void make(nether_noise* noise, int64_t seed) {
    make_nether_layer(noise, seed);
}