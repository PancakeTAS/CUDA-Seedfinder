#include "searcher.hu"

#define A 341873128712L
#define B 132897987541L

__global__ void startSearch(int64_t structureSeedOffset) {
    // Figure out what seed to check
    int64_t index = ((structureSeedOffset + threadIdx.x) + (((int64_t) blockIdx.x) * 1024));
    int64_t structureSeed = (index << 16) + structureSeedOffset;

    printf("Found structure seed: %llu\n", structureSeed);
}