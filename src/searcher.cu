#include "searcher.hu"

#define A 341873128712L
#define B 132897987541L

__global__ void startSearch(int64_t structureSeedOffset) {
    // Figure out what seed to check
    int64_t index = ((structureSeedOffset + threadIdx.x) + (((int64_t) blockIdx.x) * 1024));
    int64_t structureSeed = (index << 16) + structureSeedOffset;

    // Find the nether fortress structure on the seed
    uint64_t xz = locate_structure(structureSeed,
        // Position Seed Part
        0 * A + 0 * B, 
        // Spaced X and Z Region Coordinates
        0 * 27L,
        0 * 27L,
        // Salt and Offset
        23L, 30084232L,
        // Edge Case
        3
    );

    if (xz == 0xFFFFFFFFFFFFFFFF) // Return if the structure wasn't found
        return; 

    int32_t x = xz >> 32;
    int32_t z = xz;

    if (x > 1 && z > 1)
        return;

    

    printf("Found structure seed: %llu\n", structureSeed);
}