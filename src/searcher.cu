#include "searcher.hu"

#define A 341873128712L
#define B 132897987541L

__global__ void startSearch(cond* condition, uint32_t conditioncount, uint64_t structureSeedOffset) {
    // Figure out what seed to check
    uint64_t index = ((structureSeedOffset + threadIdx.x) + (blockIdx.x * 1024));
    uint64_t structureSeed = index << 16;

    uint32_t regionX = condition[0].regionX;
    uint32_t regionZ = condition[0].regionZ;
    uint32_t spacing = condition[0].spacing;
    uint32_t offset = condition[0].offset;
    uint64_t salt = condition[0].salt;

    // Find the first structure on that seed
    uint64_t xz = locate_structure(structureSeed,
        // Position Seed Part
        regionX * A + regionZ * B, 
        // Spaced X and Z Region Coordinates
        regionX * spacing, 
        regionZ * spacing, 
        // Salt and Offset
        offset, salt
    );
    uint32_t x = xz >> 32;
    uint32_t z = xz;

    printf("On seed %llu, the first condition was met at %d, %d.\n", structureSeed, x, z);
}