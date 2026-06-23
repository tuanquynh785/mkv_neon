// mkv-neonbs-glue.c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------
// TỐI ƯU CẤU TRÚC 128-BIT LANES (128 BYTES CONCURRENTLY)
// Sử dụng __int128 để thuật toán C chạy chuẩn 128-bit 
// đối chiếu trực tiếp với thanh ghi v0.16b của NEON.
// ---------------------------------------------------------
typedef unsigned __int128 bs_t;

typedef struct {
    __attribute__((aligned(16))) bs_t b[8]; 
} bs_block;

// API kết nối với ARM64 NEON Backend (Cần viết bằng v0.16b - v15.16b)
extern void pack_neon(const uint8_t *in, bs_block *out);
extern void unpack_neon(const bs_block *in, uint8_t *out);

extern void sbox_bs_neon(bs_block *blk);
extern void inv_sbox_bs_neon(bs_block *blk);
extern void mixword_neon(bs_block *blk);
extern void inv_mixword_neon(bs_block *blk);

extern void encrypt_neon(uint8_t* data, int num_rounds);
extern void decrypt_neon(uint8_t* data, int num_rounds);

// =========================================================
// THUẬT TOÁN C THAM CHIẾU (128-BIT) ĐỂ DEBUG ASM
// =========================================================
#define MASK_A (((bs_t)0xAAAAAAAAAAAAAAAAULL << 64) | 0xAAAAAAAAAAAAAAAAULL)
#define MASK_C (((bs_t)0xCCCCCCCCCCCCCCCCULL << 64) | 0xCCCCCCCCCCCCCCCCULL)
#define MASK_F (((bs_t)0xF0F0F0F0F0F0F0F0ULL << 64) | 0xF0F0F0F0F0F0F0F0ULL)
#define MASK_1 (((bs_t)0x1111111111111111ULL << 64) | 0x1111111111111111ULL)
#define MASK_ALL (~((bs_t)0))

void pack128_c(const uint8_t *in, bs_block *out) {
    for (int i = 0; i < 8; i++) {
        bs_t row = 0;
        for (int j = 0; j < 16; j++) {
            row |= ((bs_t)in[i + j * 8]) << (j * 8);
        }
        out->b[i] = row;
    }
    for (int i = 0; i < 3; i++) {
        int step = 1 << i;
        bs_t mask = (i == 0) ? MASK_A : (i == 1) ? MASK_C : MASK_F;
        for (int j = 0; j < 8; j += 2 * step) {
            for (int k = 0; k < step; k++) {
                bs_t t = ((out->b[j + k] >> step) ^ out->b[j + k + step]) & (mask >> step);
                out->b[j + k] ^= (t << step);
                out->b[j + k + step] ^= t;
            }
        }
    }
}

void unpack128_c(const bs_block *in, uint8_t *out) {
    bs_t t_arr[8];
    memcpy(t_arr, in->b, 128);
    for (int i = 0; i < 3; i++) {
        int step = 1 << i;
        bs_t mask = (i == 0) ? MASK_A : (i == 1) ? MASK_C : MASK_F;
        for (int j = 0; j < 8; j += 2 * step) {
            for (int k = 0; k < step; k++) {
                bs_t x = ((t_arr[j + k] >> step) ^ t_arr[j + k + step]) & (mask >> step);
                t_arr[j + k] ^= (x << step);
                t_arr[j + k + step] ^= x;
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; j++) {
            out[i + j * 8] = (uint8_t)(t_arr[i] >> (j * 8));
        }
    }
}


void test() {
    #define ITERATIONS 2000000 // 2 triệu vòng lặp

    // 128 bytes Input
    __attribute__((aligned(16))) uint8_t data_in[128] = {
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8,
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8,
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8,
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8,
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8,
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8,
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8,
        0x54, 0x06, 0xC7, 0x3A, 0x75, 0x3C, 0x73, 0xB6, 0x26, 0xB4, 0xF1, 0x01, 0xDF, 0xAC, 0x58, 0xC8
    };

    // 128 bytes Output kỳ vọng
    __attribute__((aligned(16))) uint8_t data_out[128] = {
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4,
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4,
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4,
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4,
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4,
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4,
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4,
        0x56, 0xC9, 0x96, 0x10, 0x6C, 0x5F, 0x09, 0x56, 0x51, 0xD7, 0xEF, 0x7E, 0x09, 0x1D, 0x20, 0xA4
    };						 
    
    __attribute__((aligned(16))) uint8_t OUT_FWD[128];
    __attribute__((aligned(16))) uint8_t OUT_BWD[128];
    bs_block blk;

	// Luôn luôn align 16-bytes cho NEON
    uint8_t data[128] __attribute__((aligned(16)));
    int rounds = 1; // Ví dụ 10 vòng


	// ENCRYPT
	memcpy(data, data_in, 128);
	encrypt_neon(data, rounds);

    int fwd_errors = 0;
    for(int i = 0; i < 128; i++) if(data[i] != data_out[i]) fwd_errors++;


    printf("Encrypt\n");
    if (fwd_errors > 0) {
        printf("Result: FAILED (%d errors)\n", fwd_errors);
        printf("Data thực tế sau Encrypt (128 bytes): \n");
        for(int i = 0; i < 128; i++){
            printf("0x%02X,", data[i]);
            if ((i + 1) % 16 == 0) printf("\n"); else printf(" ");
        }
    } else {
        printf("Result: PASSED\n");
    }

	// DECRYPT
	memcpy(data, data_out, 128);
	decrypt_neon(data, rounds);

	int bwd_errors = 0;
    for(int i = 0; i < 128; i++) if(data[i] != data_in[i]) fwd_errors++;


	printf("Decrypt\n");
    if (fwd_errors > 0) {
        printf("Result: FAILED (%d errors)\n", fwd_errors);
        printf("Data thực tế sau Decrypt (128 bytes): \n");
        for(int i = 0; i < 128; i++){
            printf("0x%02X,", data[i]);
            if ((i + 1) % 16 == 0) printf("\n"); else printf(" ");
        }
    } else {
        printf("Result: PASSED\n");
    }

    // =========================================================
    // 3. BENCHMARK TỐC ĐỘ CAO (128 BYTES PER ROUND)
    // =========================================================
	printf("\n");
	clock_t start, end;
	double time_spent, mbps;

	if (fwd_errors != 0 && bwd_errors == 0) {
		// Encrypt
        start = clock();
        for(int i = 0; i < ITERATIONS; i++) {
			encrypt_neon(data_in, rounds);
        }

        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;

        // 128 bytes mỗi iteration
        mbps = (128.0 * ITERATIONS) / (time_spent * 1024 * 1024);

		printf("Encrypt\n");
        printf("Thời gian chạy: %.4f giây\n", time_spent);
        printf("Tốc độ băng thông: %.2f MB/s\n", mbps);

		// Decrypt
        start = clock();
        for(int i = 0; i < ITERATIONS; i++) {
			decrypt_neon(data_in, rounds);
        }

       end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;

        // 128 bytes mỗi iteration
        mbps = (128.0 * ITERATIONS) / (time_spent * 1024 * 1024);

		printf("Decrypt\n");
        printf("Thời gian chạy: %.4f giây\n", time_spent);
        printf("Tốc độ băng thông: %.2f MB/s\n", mbps);
    } else {
        printf("\n[!] Bỏ qua Benchmark do ASM KERNEL còn lỗi logic.\n");
    }


}

int main(){
	test();
	return 0;
}