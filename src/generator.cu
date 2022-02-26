#include "generator.hu"

__global__ void generateBiomes(int64_t structure_seed) {
	// Generate Nether Noise
	nether_noise noise;
	make_nether_layer(&noise, 420L);
	// Sample 0
	printf("%f\n", sample_double_perlin(&noise.temperature, 10, 10));
}