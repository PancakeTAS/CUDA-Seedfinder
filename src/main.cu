#include "searcherss.hu"

int main() {
    // Start the search and sync up
    search_structure<<<1024*1024*1024,1>>>(0);
    cudaDeviceSynchronize();
}