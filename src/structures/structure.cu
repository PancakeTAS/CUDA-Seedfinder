#include "structure.hu"
#include <inttypes.h>

__device__ uint64_t locate_structure(int64_t structureSeed, int64_t position, int32_t spacedRegionX, int32_t spacedRegionZ, int32_t offset, int64_t salt, uint8_t edge_case) {
    int64_t seed;

    scramble(&seed, 
        // First calculate the position seed and let 'setSeed' scramble it afterwards.
        (position + structureSeed + salt)
    );

    // Generate the next int with the offset as bound twice for x and z
    int32_t x_offset = nextInt(&seed, offset);
    int32_t z_offset = nextInt(&seed, offset);

    // Now calculate the chunk position for x and z
    int64_t x = spacedRegionX + x_offset;
    int64_t z = spacedRegionZ + z_offset;

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
            next(&seed, 32);
            if (nextInt(&seed, 5) != 0)
                return 0xFFFFFFFFFFFFFFFF;
            break;
        // Third Edge Case: 
        //          Fortress have another check to pass
        case 3:
            if (nextInt(&seed, 5) >= 2)
                return 0xFFFFFFFFFFFFFFFF;
            break;
        // Fourth Edge Case: 
        //          Triangular Structures (EndCity, Mansion, Monument) are calculated differently
        case 4:
            // Do a bit of trickery with the random number generator, because it should generate 2 numbers for x and z in order.
            x += -x_offset + (x_offset + z_offset) / 2;
            z += -z_offset + (nextInt(&seed, offset) + nextInt(&seed, offset)) / 2;
            break;
        default:
            break;
    }

    return x << 32 | z;
}