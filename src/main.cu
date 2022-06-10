//#include "searcher.hu"
#include <stdio.h>
#include <stdint.h>
/*
void mix_seed(uint64_t* seed, int64_t salt) {
    *seed *= (*seed*6364136223846793005LL + 1442695040888963407LL);
    *seed += salt;
}

uint64_t get_mid_salt(uint64_t salt) {
    uint64_t midsalt = salt;
    mix_seed(&midsalt, salt);
    mix_seed(&midsalt, salt);
    mix_seed(&midsalt, salt);
    return midsalt;
}

uint64_t get_layer_seed(uint64_t world_seed, uint64_t salt) {
    uint64_t mid_salt = get_mid_salt(salt);
    uint64_t layer_seed = world_seed;
    mix_seed(&layer_seed, mid_salt);
    mix_seed(&layer_seed, mid_salt);
    mix_seed(&layer_seed, mid_salt);
    return layer_seed;
}

uint64_t get_local_seed(uint64_t layer_seed, int32_t x, int32_t z) {
    mix_seed(&layer_seed, x);
    mix_seed(&layer_seed, z);
    mix_seed(&layer_seed, x);
    mix_seed(&layer_seed, z);
    return layer_seed;
}

int32_t next_int(uint64_t* local_seed, uint64_t layer_seed, int32_t bound) {
    int64_t x = (*local_seed >> 24);
    int32_t i = x % bound;
    if ((x ^ bound) < 0 && i != 0)
        i += bound;
    mix_seed(local_seed, layer_seed);
    return i;
}

int32_t choose2(uint64_t* local_seed, uint64_t layer_seed, int32_t a, int32_t b) {
    return next_int(local_seed, layer_seed, 2) == 0 ? a : b;
}

int32_t choose4(uint64_t* local_seed, uint64_t layer_seed, int32_t a, int32_t b, int32_t c, int32_t d) {
    int32_t i = next_int(local_seed, layer_seed, 4);
    return i == 0 ? a : i == 1 ? b : i == 2 ? c : d;
}

int32_t layer2(uint32_t l1[3][3], uint64_t layer_seed, int32_t x, int32_t z) {
    int32_t l2_center = l1[x>>1][z>>1]; 
    uint64_t local_seed = get_local_seed(layer_seed, x & -2, z & -2);
    int32_t xb = x & 1;
    int32_t zb = z & 1;
    
    if (xb == 0 && zb == 0) return l2_center;
    int32_t s = l1[x>>1][(z+1)>>1];
    int32_t z_plus = choose2(&local_seed, layer_seed, l2_center, s);

    if (xb == 0) return z_plus;

    int32_t e = l1[(x+1)>>1][z>>1];
    int32_t x_plus = choose2(&local_seed, layer_seed, l2_center, e);

    if (zb == 0) return x_plus;

    return choose4(&local_seed, layer_seed, l2_center, e, s, l1[(x+1)>>1][(z+1)>>1]);
}

int32_t layer3(uint32_t l2[5][5], uint64_t layer_seed, int32_t x, int32_t z) {
    uint64_t local_seed = get_local_seed(layer_seed, x, z);

    int32_t sw = l2[x-1][z+1];
    int32_t se = l2[x+1][z+1];
    int32_t ne = l2[x+1][z-1];
    int32_t nw = l2[x-1][z-1];
    int32_t center = l2[x][z];

    return center;
}
*/

int main() {


    //uint64_t world_seed = 6;

    /* Layer 1 

    // prepare variables for layer 1
    uint32_t l1[5][5];
    uint64_t l1_salt = 1;
    uint64_t l1_layer_seed = get_layer_seed(world_seed, l1_salt);

    // apply layer 1
    for (int32_t x = 0; x < 5; x++) {
        for (int32_t z = 0; z < 5; z++) {
            uint64_t l1_local_seed = get_local_seed(l1_layer_seed, x, z);
            l1[x][z] = x == 0 && z == 0 || next_int(&l1_local_seed, l1_layer_seed, 10) == 0 ? 1 : 0;
        }
    }

    // print layer 1
    printf("Layer 1: \n%d %d %d\n%d %d %d\n%d %d %d\n", l1[0][0], l1[0][1], l1[0][2], l1[1][0], l1[1][1], l1[1][2], l1[2][0], l1[2][1], l1[2][2]);

    /* Layer 2 

    // prepare variables for layer 2
    uint32_t l2[5][5];
    uint64_t l2_salt = 2000;
    uint64_t l2_layer_seed = get_layer_seed(world_seed, l2_salt);

    // apply layer 2
    for (int32_t x = 0; x < 5; x++) {
        for (int32_t z = 0; z < 5; z++) {
            l2[x][z] = layer2(l1, l2_layer_seed, x, z);
        }
    }

    // print layer 2
    printf("Layer 2: \n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n", l2[0][0], l2[0][1], l2[0][2], l2[0][3], l2[0][4], l2[1][0], l2[1][1], l2[1][2], l2[1][3], l2[1][4], l2[2][0], l2[2][1], l2[2][2], l2[2][3], l2[2][4], l2[3][0], l2[3][1], l2[3][2], l2[3][3], l2[3][4], l2[4][0], l2[4][1], l2[4][2], l2[4][3], l2[4][4]);

    /* Layer 3 

    // prepare variables for layer 3
    uint32_t l3[5][5];
    uint64_t l3_salt = 1;
    uint64_t l3_layer_seed = get_layer_seed(world_seed, l3_salt);

    // apply layer 3
    for (int32_t x = 1; x < 4; x++) {
        for (int32_t z = 1; z < 4; z++) {
            l3[x][z] = layer3(l2, l3_layer_seed, x, z);
        }
    }

    // print layer 3
    printf("Layer 3: \n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n%d %d %d %d %d\n", l3[0][0], l3[0][1], l3[0][2], l3[0][3], l3[0][4], l3[1][0], l3[1][1], l3[1][2], l3[1][3], l3[1][4], l3[2][0], l3[2][1], l3[2][2], l3[2][3], l3[2][4], l3[3][0], l3[3][1], l3[3][2], l3[3][3], l3[3][4], l3[4][0], l3[4][1], l3[4][2], l3[4][3], l3[4][4]);

    // Start the search and sync up
    //startSearch<<<1024*1024*1024,256>>>(0);
    //cudaDeviceSynchronize();*/
}