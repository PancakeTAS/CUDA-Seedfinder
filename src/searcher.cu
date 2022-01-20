#include "searcher.hu"

#define A 341873128712L
#define B 132897987541L

__global__ void startSearch(int64_t structureSeedOffset) {
    // Figure out what seed to check
    int64_t index = ((structureSeedOffset + threadIdx.x) + (((int64_t) blockIdx.x) * 1024));
    int64_t structureSeed = (index << 16) + structureSeedOffset;

    // Check for a ruined portal at chunk 0, 0
    int64_t xz_ruinedportal = locate_structure(structureSeed, 0, 0, 0, 25, 34222645L, 0);
    int32_t x_ruinedportal = xz_ruinedportal >> 32;
    int32_t z_ruinedportal = xz_ruinedportal;

    if (x_ruinedportal > 0) 
        return;

    if (z_ruinedportal > 0)
        return;

    // Check for a bastion remnant at chunk 0, 0
    int64_t xz_bastionremnant = locate_structure(structureSeed, 0, 0, 0, 23, 30084232L, 1);
    int32_t x_bastionremnant = xz_bastionremnant >> 32;
    int32_t z_bastionremnant = xz_bastionremnant;

    if (x_bastionremnant > 0)
        return;

    if (z_bastionremnant > 0)
        return;

    // Check for a fortress at chunk -1, -1
    int64_t xz_fortress = locate_structure(structureSeed, -1 * A + -1 * B, -1 * 27, -1 * 27, 23, 30084232L, 3);
    int32_t x_fortress = xz_fortress >> 32;
    int32_t z_fortress = xz_fortress;

    if (x_fortress < -1)
        return;

    if (z_fortress < -1)
        return;
    
    // Check for a finishable ruined portal - note: this code will return as soon as something invalid is found and is not going to check for all infos
    if (!generateRuinedPortal(structureSeed, x_ruinedportal, z_ruinedportal))
        return;

    printf("Found structure seed: %llu\n", structureSeed);
}