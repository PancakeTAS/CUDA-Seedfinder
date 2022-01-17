#include "searcher.hu"

#define A 341873128712L
#define B 132897987541L

__global__ void startSearch(cond* condition, uint32_t conditioncount, uint64_t structureSeedOffset) {
    // Figure out what seed to check
    uint64_t index = ((structureSeedOffset + threadIdx.x) + (((uint64_t) blockIdx.x) * 1024));
    uint64_t structureSeed = index;

    size_t i;
    uint32_t regionX;
    uint32_t regionZ;
    uint32_t spacing;
    uint64_t xz;
    uint32_t x;
    uint32_t z;
    uint32_t chunkXMin;
    uint32_t chunkZMin;
    uint32_t chunkXMax;
    uint32_t chunkZMax;
    // Check all conditions
    for (i = 0; i < conditioncount; i++) {
        // Load condition data
        regionX = condition[i].regionX;
        regionZ = condition[i].regionZ;
        spacing = condition[i].spacing;
        chunkXMin = condition[i].chunkXMin;
        chunkZMin = condition[i].chunkZMin;
        chunkXMax = condition[i].chunkXMax;
        chunkZMax = condition[i].chunkZMax;

        // Find the first structure on that seed
        xz = locate_structure(structureSeed,
            // Position Seed Part
            regionX * A + regionZ * B, 
            // Spaced X and Z Region Coordinates
            regionX * spacing, 
            regionZ * spacing, 
            // Salt and Offset
            condition[i].offset, condition[i].salt,
            // Edge Case
            condition[i].edge_case
        );
        if (xz == 0xFFFFFFFFFFFFFFFF) // Return if the structure wasn't found
            return;

        // Relative Positioning
        if (condition[i].relativeTo) {
            chunkXMin = x - chunkXMin;
            chunkZMin = z - chunkZMin;
            chunkXMin *= !(chunkXMin > 32);
            chunkZMin *= !(chunkZMin > 32);
            chunkXMax += x;
            chunkZMax += z;
        }

        x = xz >> 32;
        z = xz;

        // Check positioning
        if (x < chunkXMin || x > chunkXMax || z < chunkZMin || z > chunkZMax)
            return;
    }
    printf("Found structure seed: %llu\n", structureSeed);
}