#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli.h"
#include "utils.h"
#include "crypto.h"
#include "keygen.h"
#include "gpu.h"

int main(int argc, char **argv) {
    cli_options opts = {0};
    if (!parse_cli(argc, argv, &opts)) {
        print_usage();
        return 1;
    }

    // Load files
    size_t enc_len = 0, orig_len = 0;
    unsigned char *enc_buf = load_file(opts.enc_path, &enc_len);
    if (!enc_buf) { fprintf(stderr, "Failed to load encrypted file\n"); return 2; }
    unsigned char *orig_buf = load_file(opts.orig_path, &orig_len);
    if (!orig_buf) { fprintf(stderr, "Failed to load original file\n"); free(enc_buf); return 3; }

    // Prepare cipher spec
    cipher_spec spec = {0};
    if (!init_cipher_spec_from_opts(&spec, &opts)) {
        fprintf(stderr, "Unsupported algo/mode or IV missing.\n");
        free(enc_buf); free(orig_buf);
        return 4;
    }

    // IV auto-detect for CBC: use first block as IV and skip it in ciphertext
    unsigned char *enc_ptr = enc_buf;
    size_t enc_len_adj = enc_len;
    if (spec.use_cbc && spec.iv_from_file) {
        if (enc_len < (size_t)spec.block_len) {
            fprintf(stderr, "Encrypted file too small to contain IV.\n");
            free(enc_buf); free(orig_buf);
            return 5;
        }
        memcpy(spec.iv, enc_buf, spec.block_len);
        enc_ptr = enc_buf + spec.block_len;
        enc_len_adj = enc_len - spec.block_len;
        fprintf(stdout, "[*] CBC IV auto-detected from file header (%d bytes)\n", spec.block_len);
    }

    // Hash original once
    unsigned char orig_hash[32];
    sha256_hash(orig_buf, orig_len, orig_hash);

    // ZIP header heuristic for .docx
    const unsigned char zip_header[4] = {0x50, 0x4B, 0x03, 0x04};

    // Keyspace spec
    keyspace_spec ks = {0};
    init_keyspace_from_opts(&ks, &opts);

    // Output buffers
    unsigned char *dec_buf = (unsigned char*)malloc(enc_len_adj);
    if (!dec_buf) {
        fprintf(stderr, "Alloc failed for dec buffer.\n");
        free(enc_buf); free(orig_buf);
        return 6;
    }

    // Progress state
    progress_state prog = {0};
    init_progress(&prog, opts.resume_file);

    // Search
    found_key fk = {0};
    int result = 0;

    if (opts.use_gpu) {
        fprintf(stdout, "[*] Starting GPU-assisted search over 32-bit segment...\n");
        result = gpu_search(enc_ptr, enc_len_adj, orig_hash, &ks, &spec, zip_header, &prog, &fk);
    } else {
        fprintf(stdout, "[*] Starting CPU OpenMP search over 32-bit segment...\n");
        result = cpu_search(enc_ptr, enc_len_adj, orig_hash, &ks, &spec, zip_header, &prog, &fk, dec_buf);
    }

    if (result == 1) {
        fprintf(stdout, "[+] Match found!\n");
        fprintf(stdout, "    Key hex: ");
        for (int i = 0; i < spec.key_len; i++) printf("%02x", fk.key[i]);
        printf("\n");

        // Decrypt full adjusted ciphertext and save
        int dec_len = decrypt_buffer(enc_ptr, (int)enc_len_adj, fk.key, spec.iv, dec_buf, &spec);
        if (dec_len <= 0) {
            fprintf(stderr, "Decryption failed when writing output\n");
        } else {
            if (opts.out_path) {
                if (!save_file(opts.out_path, dec_buf, dec_len)) {
                    fprintf(stderr, "Failed to write output file\n");
                } else {
                    fprintf(stdout, "[+] Decrypted output written to: %s\n", opts.out_path);
                }
            }
        }
    } else if (result == 0) {
        fprintf(stdout, "[-] No match found in current keyspace segment.\n");
    } else {
        fprintf(stderr, "[!] Error during search.\n");
    }

    save_progress(&prog, opts.resume_file);
    free(dec_buf);
    free(enc_buf);
    free(orig_buf);
    return (result == 1) ? 0 : ((result < 0) ? 7 : 6);
}