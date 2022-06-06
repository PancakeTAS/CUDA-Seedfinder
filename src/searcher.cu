#include "searcher.hu"

#define A 341873128712LL
#define B 132897987541LL

__global__ void startSearch(int64_t structureSeedOffset) {
    // Figure out what seed to check
    int64_t index = ((structureSeedOffset + threadIdx.x) + (((int64_t) blockIdx.x) * 256));
    int64_t structureSeed = (index << 16) + structureSeedOffset;

    for (size_t cx = 0; cx < 3; cx++) {
        size_t cz = cx;

        // Find the nether fortress structure on the seed
        uint64_t xz = locate_structure(structureSeed,
            // Position Seed Part
            cx * A + cz * B, 
            // Spaced X and Z Region Coordinates
            cx * 27L,
            cz * 27L,
            // Salt and Offset
            23L, 30084232LL,
            // Edge Case
            3
        );

        if (xz == 0xFFFFFFFFFFFFFFFF) // Return if the structure wasn't found
            return; 

        int32_t x = xz >> 32;
        int32_t z = xz;

        if (x > cx*27L+1 || z > cz*27L+1)
            return;
    }

    // Find and check for the correct biome on the seed
    nether_noise noise;
    make_nether_layer(&noise, structureSeed);
    int i = get_nether_biome(&noise, 0, 0);
    if (i != 1)
        return;

    printf("Found structure seed: %llu\n", structureSeed);
}