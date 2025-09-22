#ifndef CLI_H
#define CLI_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    const char *enc_path;
    const char *orig_path;
    const char *out_path;
    const char *algo;    // "AES256" or "DES"
    const char *mode;    // "ECB" or "CBC"
    bool use_gpu;
    const char *segment; // "first4", "last4", "middle4"
    char prefix_hex[65]; // 64 hex chars for AES-256 prefix (optional)
    char suffix_hex[65]; // 64 hex chars for AES-256 suffix (optional)
    char iv_hex[33];     // 32 hex chars for AES IV
    bool iv_auto;        // use first block of encrypted file as IV for CBC
    const char *resume_file;
    uint64_t max_keys;   // default 2^32 if 0
    uint32_t batch;      // GPU batch size (default 1M)
} cli_options;

bool parse_cli(int argc, char **argv, cli_options *out);
void print_usage(void);

#endif