#include "ruinedportal.hu"

__device__ bool generateRuinedPortal(int64_t structureSeed, int32_t x, int32_t z) {
    // IMPORTANT: DO NOT USE THIS ON THE MAIN BRANCH. 
    // This implementation currently only works for non-jungle [exception here, this works], non-desert, non-swamp, non-ocean and non-nether portals.
    
    int64_t seed;
    scrambleCarverSeed(&seed, structureSeed, x, z);

    // Check for non-buried portals
    if (nextFloat(&seed) < 0.5f)
        return false;
    
    // skip airpocket
    nextFloat(&seed);

    // Check for non-giant portals
    if (nextFloat(&seed) < 0.05f)
        return false;

    // Check for type 6 portals
    if (nextInt(&seed, 10) != 5)
        return false;

    // Check for non-rotated portals
    if (nextIntPower(&seed, 4) != 0)
        return false;

    // Check for non-mirrored portals
    if (nextFloat(&seed) >= 0.5)
        return false;

    // Check for crying obsidian
    int y_pos_obsidian[] = {1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    int z_pos_obsidian[] = {1, 2, 3, 0, 4, 0, 4, 0, 4, 1, 3};
    int y_pos, z_pos;

    // Update position seed and loop through blocks
    for (size_t i = 0; i < 11; i++) {
        y_pos = y_pos_obsidian[i];
        z_pos = z_pos_obsidian[i] + (z * 16);
        scramblePositionSeed(&seed, 2, y_pos, z_pos);

    }
    
    // 2, 3

    return true;
}