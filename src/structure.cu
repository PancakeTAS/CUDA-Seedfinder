#include "structure.hu"

__device__ uint64_t locate_structure(uint64_t structureSeed, uint32_t position, uint32_t spacedRegionX, uint32_t spacedRegionZ, uint32_t offset, uint64_t salt) {
    uint64_t seed;
    scramble(&seed, 
        // First calculate the position seed and let 'setSeed' scramble it afterwards.
        (position + structureSeed + salt)
    );

    // Generate the next int with the offset as bound twice for x and z
    uint32_t x_offset = nextInt(&seed, offset);
    uint32_t z_offset = nextInt(&seed, offset);

    // Now calculate the chunk position for x and z
    uint64_t x = spacedRegionX + x_offset;
    uint64_t z = spacedRegionZ + z_offset;

    return x << 32 | z;
}