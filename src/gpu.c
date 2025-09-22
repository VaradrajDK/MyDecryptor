#include "gpu.h"
#include "cli.h"
#include "keygen.h"
#include "aes_tables.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

extern "C" int load_aes_tables_to_device(const uint8_t *sbox, const uint8_t *inv);
extern "C" void launch_try_batch_stream(const unsigned char *d_enc_block,
                                        const keyspace_spec *d_ks,
                                        const cipher_spec *d_spec,
                                        uint64_t start_idx, uint32_t batch,
                                        unsigned char *d_out_keys,
                                        uint32_t *d_out_count,
                                        cudaStream_t stream);

static int ensure_tables_loaded(void) {
    static int loaded = 0;
    if (loaded) return 0;
    int rc = load_aes_tables_to_device(AES_SBOX, AES_INV_SBOX);
    if (rc != 0) return rc;
    loaded = 1;
    return 0;
}

int cpu_search(const unsigned char *enc, size_t enc_len, const unsigned char orig_hash[32],
               const keyspace_spec *ks, const cipher_spec *spec, const unsigned char zip_header[4],
               progress_state *prog, found_key *out, unsigned char *dec_buf_shared) {

    unsigned char iv[16]; memcpy(iv, spec->iv, 16);

    uint64_t total = ks->max;
    uint64_t start = prog->tried;

    volatile int found = 0;

    #pragma omp parallel
    {
        unsigned char key[32] = {0};
        unsigned char blk_plain[32] = {0};
        unsigned char blk_ciph[32] = {0};
        unsigned char *dec_buf = (unsigned char*)malloc(enc_len);
        if (!dec_buf) dec_buf = dec_buf_shared;

        #pragma omp for schedule(dynamic, 4096) nowait
        for (uint64_t i=start; i<total; i++) {
            if (found) continue;

            construct_key_from_index((uint32_t)i, key, ks, spec);

            memcpy(blk_ciph, enc, spec->block_len);
            int bl = decrypt_block_raw_ecb(blk_ciph, key, blk_plain, spec);
            if (bl > 0) {
                if (spec->use_cbc) {
                    for (int j=0;j<spec->block_len;j++) blk_plain[j] ^= iv[j];
                }
                if (bl >= 4 && blk_plain[0]==zip_header[0] && blk_plain[1]==zip_header[1] &&
                    blk_plain[2]==zip_header[2] && blk_plain[3]==zip_header[3]) {
                    int dec_len = decrypt_buffer(enc, (int)enc_len, key, iv, dec_buf, spec);
                    if (dec_len > 0) {
                        unsigned char h[32]; sha256_hash(dec_buf, dec_len, h);
                        if (memeq(h, orig_hash, 32)) {
                            #pragma omp critical
                            { if (!found) { memcpy(out->key, key, spec->key_len); found = 1; } }
                        }
                    }
                }
            }

            if ((i & ((1<<22)-1)) == 0 && omp_get_thread_num()==0) {
                prog->tried = i+1;
                fprintf(stdout, "[CPU-OMP] Progress: %llu / %llu\n",
                        (unsigned long long)(i+1), (unsigned long long)total);
            }
        }

        if (dec_buf != dec_buf_shared) free(dec_buf);
    }

    return found ? 1 : 0;
}

int gpu_search(const unsigned char *enc, size_t enc_len, const unsigned char orig_hash[32],
               const keyspace_spec *ks, const cipher_spec *spec, const unsigned char zip_header[4],
               progress_state *prog, found_key *out) {

    if (spec->is_aes == 0) {
        fprintf(stderr, "[GPU] Device heuristic currently supports AES-256 only; using CPU.\n");
        return 0;
    }
    if (ensure_tables_loaded() != 0) {
        fprintf(stderr, "[GPU] Failed to load AES tables to device.\n");
        return -1;
    }

    const uint32_t batch = 1048576; // 1M
    uint64_t total = ks->max;
    uint64_t start = prog->tried;

    // Device buffers reused across launches
    unsigned char *d_enc_block = nullptr;
    keyspace_spec *d_ks = nullptr;
    cipher_spec *d_spec = nullptr;

    cudaMalloc(&d_enc_block, spec->block_len);
    cudaMemcpy(d_enc_block, enc, spec->block_len, cudaMemcpyHostToDevice);

    cudaMalloc(&d_ks, sizeof(keyspace_spec));
    cudaMemcpy(d_ks, ks, sizeof(keyspace_spec), cudaMemcpyHostToDevice);

    cudaMalloc(&d_spec, sizeof(cipher_spec));
    cudaMemcpy(d_spec, spec, sizeof(cipher_spec), cudaMemcpyHostToDevice);

    // Double-buffered output and streams
    unsigned char *d_out_keys[2] = {nullptr, nullptr};
    uint32_t *d_out_count[2] = {nullptr, nullptr};
    unsigned char *h_out_keys[2] = {nullptr, nullptr};
    uint32_t h_out_count[2] = {0, 0};
    cudaStream_t streams[2];

    for (int i=0;i<2;i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_out_keys[i], batch * spec->key_len);
        cudaMalloc(&d_out_count[i], sizeof(uint32_t));
        cudaHostAlloc((void**)&h_out_keys[i], batch * spec->key_len, cudaHostAllocDefault);
    }

    int buf = 0;
    while (start < total) {
        uint32_t zero = 0;
        cudaMemcpyAsync(d_out_count[buf], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, streams[buf]);

        // Launch kernel on this stream
        launch_try_batch_stream(d_enc_block, d_ks, d_spec, start, batch,
                                d_out_keys[buf], d_out_count[buf], streams[buf]);

        // Async copy device results to host (full keys buffer; count separately)
        cudaMemcpyAsync(&h_out_count[buf], d_out_count[buf], sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, streams[buf]);
        cudaMemcpyAsync(h_out_keys[buf], d_out_keys[buf],
                        batch * spec->key_len, cudaMemcpyDeviceToHost, streams[buf]);

        // Process previous buffer while current stream is running
        int prev = buf ^ 1;
        cudaStreamSynchronize(streams[prev]);

        uint32_t prev_found = h_out_count[prev];
        for (uint32_t i=0; i<prev_found; i++) {
            const unsigned char *key = h_out_keys[prev] + i*spec->key_len;
            unsigned char *dec = (unsigned char*)malloc(enc_len);
            int dec_len = decrypt_buffer(enc, (int)enc_len, key, spec->iv, dec, spec);
            if (dec_len > 0) {
                unsigned char h[32]; sha256_hash(dec, dec_len, h);
                if (memeq(h, orig_hash, 32)) {
                    memcpy(out->key, key, spec->key_len);
                    free(dec);
                    // Cleanup
                    cudaFree(d_enc_block);
                    cudaFree(d_ks);
                    cudaFree(d_spec);
                    for (int k=0;k<2;k++) {
                        cudaStreamDestroy(streams[k]);
                        cudaFree(d_out_keys[k]);
                        cudaFree(d_out_count[k]);
                        cudaFreeHost(h_out_keys[k]);
                    }
                    return 1;
                }
            }
            free(dec);
        }

        start += batch;
        prog->tried = start;
        fprintf(stdout, "[GPU] Progress: %llu / %llu keys\n",
                (unsigned long long)start, (unsigned long long)total);
        buf ^= 1;
    }

    // Final buffer processing
    cudaStreamSynchronize(streams[buf]);
    uint32_t last_found = h_out_count[buf];
    for (uint32_t i=0; i<last_found; i++) {
        const unsigned char *key = h_out_keys[buf] + i*spec->key_len;
        unsigned char *dec = (unsigned char*)malloc(enc_len);
        int dec_len = decrypt_buffer(enc, (int)enc_len, key, spec->iv, dec, spec);
        if (dec_len > 0) {
            unsigned char h[32]; sha256_hash(dec, dec_len, h);
            if (memeq(h, orig_hash, 32)) {
                memcpy(out->key, key, spec->key_len);
                free(dec);
                cudaFree(d_enc_block);
                cudaFree(d_ks);
                cudaFree(d_spec);
                for (int k=0;k<2;k++) {
                    cudaStreamDestroy(streams[k]);
                    cudaFree(d_out_keys[k]);
                    cudaFree(d_out_count[k]);
                    cudaFreeHost(h_out_keys[k]);
                }
                return 1;
            }
        }
        free(dec);
    }

    cudaFree(d_enc_block);
    cudaFree(d_ks);
    cudaFree(d_spec);
    for (int k=0;k<2;k++) {
        cudaStreamDestroy(streams[k]);
        cudaFree(d_out_keys[k]);
        cudaFree(d_out_count[k]);
        cudaFreeHost(h_out_keys[k]);
    }
    return 0;
}