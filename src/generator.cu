#include "generator.hu"

__global__ void generateBiomes(int64_t structure_seed) {
	nether_noise noise;
	for(size_t seed = 0; seed < 1000000; seed+=100000) {
		make_nether_layer(&noise, seed);
		for(int x = 0; x < 10; x++) {
			for(int z = 0; z < 10; z++) {
				printf("%d\n", get_nether_biome(&noise, x, z));
			}	
		}	
	}
}