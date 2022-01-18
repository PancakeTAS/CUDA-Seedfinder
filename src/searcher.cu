#include "searcher.hu"

#define A 341873128712L
#define B 132897987541L

__global__ void startSearch(cond* condition, uint32_t conditioncount, int64_t structureSeedOffset) {
    // Figure out what seed to check
    int64_t index = ((structureSeedOffset + threadIdx.x) + (((int64_t) blockIdx.x) * 1024));
    int64_t structureSeed = (index << 16) + structureSeedOffset;

    size_t i;
    int32_t regionX;
    int32_t regionZ;
    int32_t spacing;
    uint64_t xz;
    int32_t x;
    int32_t z;
    int32_t chunkXMin;
    int32_t chunkZMin;
    int32_t chunkXMax;
    int32_t chunkZMax;
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