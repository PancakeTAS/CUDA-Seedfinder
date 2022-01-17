#include "searcher.hu"

int main() {
    cond conditions[1];
    
    // Create a testing condition, that searches for a pillager outpost at spawn within a 4 chunk radius towards pos pos
    cond condition1;
    condition1.regionX = 0;
    condition1.regionZ = 0;
    condition1.chunkXMin = 0;
    condition1.chunkZMin = 0;
    condition1.chunkXMax = 4;
    condition1.chunkZMax = 4;
    condition1.relativeTo = 0;
    condition1.offset = 27L;
    condition1.spacing = 32L;
    condition1.salt = 10387313L;
    condition1.edge_case = 4;

    conditions[0] = condition1;

    // Load the conditions to the GPU
    cond* gpu_conditions;
    cudaMalloc(&gpu_conditions, 1 * sizeof(cond));
    cudaMemcpy(gpu_conditions, &conditions, 1 * sizeof(cond), cudaMemcpyHostToDevice);

    // Start the search and sync up
    startSearch<<<1024*1024,1024>>>(gpu_conditions, 1, 0);
    cudaDeviceSynchronize();

    // Free the conditions array from the gpu
    cudaFree(gpu_conditions);
}

/*
#define RUINEDPORTAL_SALT 34222645L
#define RUINEDPORTAL_SPACING 40
#define RUINEDPORTAL_SEPARATION 15
#define RUINEDPORTAL_OFFSET 25
#define VILLAGE_SALT 10387312L
#define VILLAGE_SPACING 32
#define VILLAGE_SEPARATION 8
#define VILLAGE_OFFSET 24

__global__ void search(uint32_t position, uint32_t spacedRegionX_ruinedportal, uint64_t spacedRegionZ_ruinedportal, uint64_t spacedRegionX_village, uint64_t spacedRegionZ_village, uint64_t structureSeedOffset) {
    uint64_t index = ((structureSeedOffset + threadIdx.x) + (blockIdx.x * 1024));
    uint64_t structureSeed = index << 16;

    uint64_t xz_ruinedportal = find_structure(structureSeed, position, spacedRegionX_ruinedportal, spacedRegionZ_ruinedportal, RUINEDPORTAL_OFFSET, RUINEDPORTAL_SALT);
    uint32_t x_ruinedportal = xz_ruinedportal >> 32;
    uint32_t z_ruinedportal = xz_ruinedportal;

    uint64_t xz_village = find_structure(structureSeed, position, spacedRegionX_village, spacedRegionZ_village, VILLAGE_OFFSET, VILLAGE_SALT);
    uint32_t x_village = xz_village >> 32;
    uint32_t z_village = xz_village;

    if ((x_ruinedportal + z_ruinedportal + x_village + z_village) < 10) {
        printf("%d, %d - Ruined Portal; %d, %d - Village: %llu\n",
            x_ruinedportal,
            z_ruinedportal,
            x_village,
            z_village,
            structureSeed
        );
    }
}

void startSearch(uint32_t regionX, uint32_t regionZ) {
    // Prepare position seed
    uint32_t position = regionX * A + regionZ * B;
    // Prepare spaced regions for ruined portals
    uint32_t spacedRegionX_ruinedportal = regionX * RUINEDPORTAL_SPACING;
    uint32_t spacedRegionZ_ruinedportal = regionZ * RUINEDPORTAL_SPACING;
    // Prepare spaced regions for villages
    uint32_t spacedRegionX_village = regionX * VILLAGE_SPACING;
    uint32_t spacedRegionZ_village = regionZ * VILLAGE_SPACING;

    // Run portal finder on the gpu
    search<<<1024*1024*1024,1024>>>(position, spacedRegionX_ruinedportal, spacedRegionZ_ruinedportal, spacedRegionX_village, spacedRegionZ_village, 0);
}

int main() {
    startSearch(0, 0);
    cudaDeviceSynchronize();
    return 0;
}*/