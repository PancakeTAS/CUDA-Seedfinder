#include "searcher.hu"

int main() {
    // Start the search and sync up
    startSearch<<<1024*1024*1024,1>>>(0);
    cudaDeviceSynchronize();
}