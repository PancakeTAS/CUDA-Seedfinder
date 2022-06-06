#include "searcher.hu"

#define A 341873128712LL
#define B 132897987541LL

__global__ void startSearch(int64_t structureSeedOffset) {
    // Figure out what seed to check
    int64_t index = ((structureSeedOffset + threadIdx.x) + (((int64_t) blockIdx.x) * 256));
    int64_t structureSeed = (index << 16) + structureSeedOffset;

    // Find the nether fortress structure on the seed
    uint64_t xz = locate_structure(structureSeed,
        // Position Seed Part
        0 * A + 0 * B, 
        // Spaced X and Z Region Coordinates
        0 * 27L,
        0 * 27L,
        // Salt and Offset
        23L, 30084232LL,
        // Edge Case
        3
    );

    if (xz == 0xFFFFFFFFFFFFFFFF) // Return if the structure wasn't found
        return; 

    int32_t x = xz >> 32;
    int32_t z = xz;

    if (x > 1 || z > 1)
        return;

    // Find and check for the correct biome on the seed
    nether_noise noise;
    make_nether_layer(&noise, structureSeed);
    int i = get_nether_biome(&noise, 0, 0);
    if (i != 1)
        return;

    printf("Found structure seed: %llu\n", structureSeed);
}