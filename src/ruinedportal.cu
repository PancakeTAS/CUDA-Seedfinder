#include "ruinedportal.hu"

// UNFINISHED & UNTESTED
__device__ bool generateRuinedPortal(int64_t structureSeed, int32_t x, int32_t z) {
    // IMPORTANT: DO NOT USE THIS ON THE MAIN BRANCH. 
    // This implementation currently only works for non-jungle [exception here, this works], non-desert, non-swamp, non-ocean and non-nether portals.
    
    int64_t seed;
    scrambleCarverSeed(&seed, structureSeed, x, z);

    // Check for non-buried portals
    if (nextFloat(&seed) < 0.5f)
        return false;
    
    // Check for non-giant portals
    if (nextFloat(&seed) < 0.05f)
        return false;

    // Check for type 6 portals
    if (nextInt(&seed, 10) != 6)
        return false;

    // skip rotation, mirror
    nextInt(&seed, 4);
    nextFloat(&seed);

    return true;
}