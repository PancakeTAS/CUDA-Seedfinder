#include "searcher.hu"

int main() {
    cond conditions[3];
    
    // Create a testing condition, that searches for a ruined portal at 0 0
    cond condition1;
    condition1.regionX = 0;
    condition1.regionZ = 0;
    condition1.chunkXMin = 0;
    condition1.chunkZMin = 0;
    condition1.chunkXMax = 1;
    condition1.chunkZMax = 1;
    condition1.relativeTo = false;
    condition1.offset = 25L;
    condition1.spacing = 40L;
    condition1.salt = 34222645L;
    condition1.edge_case = 0;

    // Create a second testing condition, that searches for a bastion near the ruined portal in a radius of 1 chunk
    cond condition2;
    condition2.regionX = 0;
    condition2.regionZ = 0;
    condition2.chunkXMin = 0;
    condition2.chunkZMin = 0;
    condition2.chunkXMax = 4;
    condition2.chunkZMax = 4;
    condition2.relativeTo = false;
    condition2.offset = 23L;
    condition2.spacing = 27L;
    condition2.salt = 30084232L;
    condition2.edge_case = 1;

    // Create a third testing condition, that searches for a fortress near the ruined portal in a radius of 1 chunk
    cond condition3;
    condition3.regionX = -1;
    condition3.regionZ = -1;
    condition3.chunkXMin = -6;
    condition3.chunkZMin = -6;
    condition3.chunkXMax = 0;
    condition3.chunkZMax = 0;
    condition3.relativeTo = false;
    condition3.offset = 23L;
    condition3.spacing = 27L;
    condition3.salt = 30084232L;
    condition3.edge_case = 3;

    conditions[0] = condition1;
    conditions[1] = condition2;
    conditions[2] = condition3;

    // Load the conditions to the GPU
    cond* gpu_conditions;
    cudaMalloc(&gpu_conditions, 3 * sizeof(cond));
    cudaMemcpy(gpu_conditions, &conditions, 3 * sizeof(cond), cudaMemcpyHostToDevice);

    // Start the search and sync up
    startSearch<<<1024*1024*1024,1>>>(gpu_conditions, 3, 0);
    cudaDeviceSynchronize();

    // Free the conditions array from the gpu
    cudaFree(gpu_conditions);
}