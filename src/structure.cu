#include "structure.hu"

__device__ uint64_t locate_structure(uint64_t structureSeed, uint32_t position, uint32_t spacedRegionX, uint32_t spacedRegionZ, uint32_t offset, uint64_t salt, uint8_t edge_case) {
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

    // Lastly calculate all edge cases
    switch (edge_case) {
        // First Edge Case: 
        //          Bastion Remnant have another check to pass
        case 1:
            if (nextInt(&seed, 5) < 2)
                return 0xFFFFFFFFFFFFFFFF;
            break;
        // Second Edge Case:
        //          Pillager Outpost have another check to pass involving a weaker seed
        case 2:
            scrambleWeakSeed(&seed, structureSeed, x, z);
            nextInt(&seed, 32);
            if (nextInt(&seed, 5) != 0)
                return 0xFFFFFFFFFFFFFFFF;
            break;
        default:
            break;
    }

    return x << 32 | z;
}