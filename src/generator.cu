#include "generator.hu"

__global__ void generateBiomes(int64_t structure_seed) {
	// Generate Nether Noise
	nether_noise noise;
	make_nether_layer(&noise, structure_seed);
}