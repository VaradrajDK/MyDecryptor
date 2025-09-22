#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include "crypto.h"
#include "keygen.h"

// Constant memory tables
__constant__ uint8_t d_sbox[256];
__constant__ uint8_t d_inv_sbox[256];

extern "C" int load_aes_tables_to_device(const uint8_t *sbox, const uint8_t *inv) {
    cudaError_t e1 = cudaMemcpyToSymbol(d_sbox, sbox, 256);
    if (e1 != cudaSuccess) return -1;
    cudaError_t e2 = cudaMemcpyToSymbol(d_inv_sbox, inv, 256);
    if (e2 != cudaSuccess) return -2;
    return 0;
}

__device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return (uint8_t)((x<<1) ^ ((x & 0x80) ? 0x1B : 0x00));
}

__device__ uint32_t sub_word(uint32_t w) {
    return (d_sbox[(w >> 24) & 0xFF] << 24) |
           (d_sbox[(w >> 16) & 0xFF] << 16) |
           (d_sbox[(w >> 8)  & 0xFF] << 8)  |
           (d_sbox[(w)       & 0xFF]);
}

__device__ uint32_t rot_word(uint32_t w) {
    return ((w << 8) | (w >> 24));
}

__device__ void key_expansion_aes256(const uint8_t key[32], uint32_t rk[60]) {
    #pragma unroll
    for (int i=0; i<8; ++i)
        rk[i] = (key[4*i]<<24) | (key[4*i+1]<<16) | (key[4*i+2]<<8) | key[4*i+3];

    uint8_t rcon = 0x01;
    for (int i=8; i<60; ++i) {
        uint32_t temp = rk[i-1];
        if (i % 8 == 0) { temp = sub_word(rot_word(temp)) ^ ((uint32_t)rcon << 24); rcon = (uint8_t)((rcon<<1) ^ ((rcon & 0x80) ? 0x1B : 0x00)); }
        else if (i % 8 == 4) { temp = sub_word(temp); }
        rk[i] = rk[i-8] ^ temp;
    }
}

__device__ void inv_shift_rows(uint8_t s[16]) {
    uint8_t t;
    t = s[13]; s[13]=s[9]; s[9]=s[5]; s[5]=s[1]; s[1]=t;
    t = s[2];  s[2]=s[10]; s[10]=t;  t=s[6];  s[6]=s[14]; s[14]=t;
    t = s[3];  s[3]=s[7];  s[7]=s[11]; s[11]=s[15]; s[15]=t;
}

__device__ void inv_sub_bytes(uint8_t s[16]) { #pragma unroll for (int i=0;i<16;i++) s[i] = d_inv_sbox[s[i]]; }

__device__ void add_round_key(uint8_t s[16], const uint32_t rk[4]) {
    const uint8_t *rkb = (const uint8_t*)rk;
    #pragma unroll
    for (int i=0;i<16;i++) s[i] ^= rkb[i];
}

__device__ uint8_t xt(uint8_t x){ return (uint8_t)((x<<1) ^ ((x&0x80)?0x1B:0)); }
__device__ void inv_mix_columns(uint8_t s[16]) {
    #pragma unroll
    for (int c=0;c<4;c++) {
        int i=c*4; uint8_t a=s[i], b=s[i+1], d=s[i+2], e=s[i+3];
        uint8_t a2=xt(a), b2=xt(b), d2=xt(d), e2=xt(e);
        uint8_t a4=xt(a2), b4=xt(b2), d4=xt(d2), e4=xt(e2);
        uint8_t a8=xt(a4), b8=xt(b4), d8=xt(d4), e8=xt(e4);
        uint8_t a9=a8^a, b9=b8^b, d9=d8^d, e9=e8^e;
        s[i]   = (uint8_t)(a9 ^ b4 ^ d2 ^ e);
        s[i+1] = (uint8_t)(a ^ b9 ^ d4 ^ e2);
        s[i+2] = (uint8_t)(a2 ^ b ^ d9 ^ e4);
        s[i+3] = (uint8_t)(a4 ^ b2 ^ d ^ e9);
    }
}

__device__ void aes256_decrypt_block(const uint8_t in[16], uint8_t out[16], const uint8_t key[32]) {
    uint32_t rk[60];
    key_expansion_aes256(key, rk);

    uint8_t s[16];
    #pragma unroll
    for(int i=0;i<16;i++) s[i]=in[i];

    add_round_key(s, &rk[56]);
    for(int round=13; round>=1; --round){
        inv_shift_rows(s); inv_sub_bytes(s); add_round_key(s, &rk[round*4]); inv_mix_columns(s);
    }
    inv_shift_rows(s); inv_sub_bytes(s); add_round_key(s, &rk[0]);

    #pragma unroll
    for(int i=0;i<16;i++) out[i]=s[i];
}

__device__ void device_construct_key(uint32_t idx, uint8_t *key,
                                     const keyspace_spec *ks, const cipher_spec *spec) {
    for (int i=0;i<spec->key_len;i++) key[i] = ks->prefix[i];
    uint8_t segment[4] = {
        (uint8_t)((idx >> 24) & 0xFF),
        (uint8_t)((idx >> 16) & 0xFF),
        (uint8_t)((idx >> 8) & 0xFF),
        (uint8_t)(idx & 0xFF)
    };
    if (ks->segment_mode == 0) {
        for (int i=0;i<4;i++) key[i] = segment[i];
    } else if (ks->segment_mode == 1) {
        for (int i=0;i<4;i++) key[spec->key_len-4+i] = segment[i];
    } else {
        int mid = spec->key_len/2 - 2;
        for (int i=0;i<4;i++) key[mid+i] = segment[i];
    }
    for (int i=0;i<spec->key_len;i++) if (ks->suffix[i] != 0) key[i] = ks->suffix[i];
}

__global__ void kernel_try_batch(const uint8_t *enc_block, const keyspace_spec *ks,
                                 const cipher_spec *spec, uint64_t start_idx, uint32_t batch,
                                 uint8_t *out_keys, uint32_t *out_count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch) return;

    uint32_t idx = (uint32_t)(start_idx + tid);
    uint8_t key[32];
    device_construct_key(idx, key, ks, spec);

    // AES-256 ECB heuristic on first block
    uint8_t out[16];
    aes256_decrypt_block(enc_block, out, key);

    if (out[0]==0x50 && out[1]==0x4B && out[2]==0x03 && out[3]==0x04) {
        uint32_t pos = atomicAdd(out_count, 1);
        uint8_t *dst = out_keys + pos * spec->key_len;
        for (int i=0;i<spec->key_len;i++) dst[i] = key[i];
    }
}

extern "C" void launch_try_batch_stream(const unsigned char *d_enc_block,
                                        const keyspace_spec *d_ks,
                                        const cipher_spec *d_spec,
                                        uint64_t start_idx, uint32_t batch,
                                        unsigned char *d_out_keys,
                                        uint32_t *d_out_count,
                                        cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((batch + block.x - 1)/block.x);
    kernel_try_batch<<<grid, block, 0, stream>>>(d_enc_block, d_ks, d_spec, start_idx, batch, d_out_keys, d_out_count);
}