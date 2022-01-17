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

    // Create a third testing condition, that searches for a monument near the ruined portal in a radius of 1 chunk
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
    startSearch<<<1024*1024*10,1024>>>(gpu_conditions, 3, 0);
    cudaDeviceSynchronize();

    // Free the conditions array from the gpu
    cudaFree(gpu_conditions);
}