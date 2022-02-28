#include "generator.hu"

__global__ void generateBiomes(int64_t structure_seed) {
	// Generate Nether Noise
	nether_noise noise;
	make_nether_layer(&noise, 420L);
	// Sample 0
	printf("%d\n", get_nether_biome(&noise, 10, 10));
}