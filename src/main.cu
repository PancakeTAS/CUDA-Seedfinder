#include "searcher.hu"

int main() {
    cond conditions[3];
    
    // Create a testing condition, that searches for a ruined portal at 0 0
    cond condition1;
    condition1.regionX = 0;
    condition1.regionZ = 0;
    condition1.chunkXMin = 0;
    condition1.chunkZMin = 0;
    condition1.chunkXMax = 4;
    condition1.chunkZMax = 4;
    condition1.relativeTo = false;
    condition1.offset = 25L;
    condition1.spacing = 40L;
    condition1.salt = 34222645L;
    condition1.edge_case = 0;

    // Create a second testing condition, that searches for a village near the ruined portal in a radius of 1 chunk
    cond condition2;
    condition2.regionX = 0;
    condition2.regionZ = 0;
    condition2.chunkXMin = 2; // the signs will automatically be swapped for the 'min' values
    condition2.chunkZMin = 2;
    condition2.chunkXMax = 2;
    condition2.chunkZMax = 2;
    condition2.relativeTo = true;
    condition2.offset = 24L;
    condition2.spacing = 32L;
    condition2.salt = 10387312L;
    condition2.edge_case = 0;

    // Create a third testing condition, that searches for a pillager outpost near the ruined portal in a radius of 1 chunk
    cond condition3;
    condition3.regionX = 0;
    condition3.regionZ = 0;
    condition3.chunkXMin = 2; // the signs will automatically be swapped for the 'min' values
    condition3.chunkZMin = 2;
    condition3.chunkXMax = 2;
    condition3.chunkZMax = 2;
    condition3.relativeTo = true;
    condition3.offset = 27L;
    condition3.spacing = 32L;
    condition3.salt = 10387313L;
    condition3.edge_case = 4;

    conditions[0] = condition1;
    conditions[1] = condition2;
    conditions[2] = condition3;

    // Load the conditions to the GPU
    cond* gpu_conditions;
    cudaMalloc(&gpu_conditions, 3 * sizeof(cond));
    cudaMemcpy(gpu_conditions, &conditions, 3 * sizeof(cond), cudaMemcpyHostToDevice);

    // Start the search and sync up
    startSearch<<<1024*1024,1024>>>(gpu_conditions, 3, 0);
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