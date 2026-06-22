// mkv_neon_cipher.S
// Pure ARM64 NEON Bitslice Implementation (Optimized for 8 Blocks of 128-bit)
    .text
    .align 4

// =========================================================
// MACROS CHO PACK/UNPACK (Đã nâng cấp lên 128-bit Lane Width)
// =========================================================
.macro bfly_step vA, vB, vMask, step, vTmp1, vTmp2
    ushr \vTmp1\().2d, \vA\().2d, #\step
    eor  \vTmp1\().16b, \vTmp1\().16b, \vB\().16b
    and  \vTmp1\().16b, \vTmp1\().16b, \vMask\().16b
    shl  \vTmp2\().2d, \vTmp1\().2d, #\step
    eor  \vA\().16b, \vA\().16b, \vTmp2\().16b
    eor  \vB\().16b, \vB\().16b, \vTmp1\().16b
.endm

.macro transpose_8x8_bytes_128 r0, r1, r2, r3, r4, r5, r6, r7
    trn1 v16.16b, \r0\().16b, \r1\().16b
    trn2 v17.16b, \r0\().16b, \r1\().16b
    trn1 v18.16b, \r2\().16b, \r3\().16b
    trn2 v19.16b, \r2\().16b, \r3\().16b
    trn1 v20.16b, \r4\().16b, \r5\().16b
    trn2 v21.16b, \r4\().16b, \r5\().16b
    trn1 v22.16b, \r6\().16b, \r7\().16b
    trn2 v23.16b, \r6\().16b, \r7\().16b
    
    trn1 \r0\().8h, v16.8h, v18.8h
    trn2 \r2\().8h, v16.8h, v18.8h
    trn1 \r1\().8h, v17.8h, v19.8h
    trn2 \r3\().8h, v17.8h, v19.8h
    trn1 \r4\().8h, v20.8h, v22.8h
    trn2 \r6\().8h, v20.8h, v22.8h
    trn1 \r5\().8h, v21.8h, v23.8h
    trn2 \r7\().8h, v21.8h, v23.8h
    
    trn1 v16.4s, \r0\().4s, \r4\().4s
    trn2 v20.4s, \r0\().4s, \r4\().4s
    trn1 v17.4s, \r1\().4s, \r5\().4s
    trn2 v21.4s, \r1\().4s, \r5\().4s
    trn1 v18.4s, \r2\().4s, \r6\().4s
    trn2 v22.4s, \r2\().4s, \r6\().4s
    trn1 v19.4s, \r3\().4s, \r7\().4s
    trn2 v23.4s, \r3\().4s, \r7\().4s
    
    mov \r0\().16b, v16.16b
    mov \r1\().16b, v17.16b
    mov \r2\().16b, v18.16b
    mov \r3\().16b, v19.16b
    mov \r4\().16b, v20.16b
    mov \r5\().16b, v21.16b
    mov \r6\().16b, v22.16b
    mov \r7\().16b, v23.16b
.endm

// =========================================================
// MACROS CHO S-BOX GF(16) (Nâng cấp toàn bộ lên .16b)
// =========================================================
.macro map_M o3, o2, o1, o0, i3, i2, i1, i0
    eor \o3\().16b, \i2\().16b, \i1\().16b
    eor \o2\().16b, \i3\().16b, \o3\().16b
    eor \o1\().16b, \i2\().16b, \i0\().16b
    eor \o0\().16b, \o3\().16b, \i0\().16b
.endm

.macro map_M_inv o3, o2, o1, o0, z3, z2, z1, z0
    eor \o3\().16b, \z3\().16b, \z2\().16b
    eor \o1\().16b, \z1\().16b, \z0\().16b
    eor \o0\().16b, \z3\().16b, \z0\().16b
    eor \o2\().16b, \z3\().16b, \o1\().16b
.endm

.macro g4_mul r1, r0, x1, x0, y1, y0, t1, t2
    eor \t1\().16b, \x1\().16b, \x0\().16b
    eor \t2\().16b, \y1\().16b, \y0\().16b
    and \t1\().16b, \t1\().16b, \t2\().16b
    and \r1\().16b, \x1\().16b, \y1\().16b
    eor \r1\().16b, \r1\().16b, \t1\().16b
    and \r0\().16b, \x0\().16b, \y0\().16b
    eor \r0\().16b, \r0\().16b, \t1\().16b
.endm

.macro gf16_sqr r3, r2, r1, r0, x3, x2, x1, x0, t1
    mov \r3\().16b, \x2\().16b
    mov \r2\().16b, \x3\().16b
    eor \t1\().16b, \x2\().16b, \x3\().16b
    eor \r1\().16b, \x0\().16b, \t1\().16b
    eor \r0\().16b, \x1\().16b, \x2\().16b
.endm

.macro gf16_mul r3, r2, r1, r0, a3, a2, a1, a0, b3, b2, b1, b0, t1, t2, t3, t4, t5, t6, t7, t8
    g4_mul \t1, \t2, \a3, \a2, \b3, \b2, \t3, \t4
    g4_mul \t5, \t6, \a1, \a0, \b1, \b0, \t3, \t4
    eor \t7\().16b, \a3\().16b, \a1\().16b
    eor \t8\().16b, \a2\().16b, \a0\().16b
    eor \r3\().16b, \b3\().16b, \b1\().16b
    eor \r2\().16b, \b2\().16b, \b0\().16b
    g4_mul \r3, \r2, \t7, \t8, \r3, \r2, \t3, \t4
    eor \r3\().16b, \r3\().16b, \t5\().16b
    eor \r2\().16b, \r2\().16b, \t6\().16b
    eor \t3\().16b, \t1\().16b, \t2\().16b
    eor \r1\().16b, \t5\().16b, \t3\().16b
    eor \r0\().16b, \t6\().16b, \t1\().16b
.endm

.macro gf16_inv r3, r2, r1, r0, x3, x2, x1, x0, t1, t2, t3, t4, dh, dl
    g4_mul \t1, \t2, \x3, \x2, \x1, \x0, \t3, \t4
    eor \dh\().16b, \x2\().16b, \x3\().16b
    eor \dh\().16b, \dh\().16b, \t1\().16b
    eor \dh\().16b, \dh\().16b, \x0\().16b
    eor \dl\().16b, \x2\().16b, \t2\().16b
    eor \dl\().16b, \dl\().16b, \x1\().16b
    g4_mul \r3, \r2, \dl, \dh, \x3, \x2, \t1, \t2
    eor \t3\().16b, \x3\().16b, \x1\().16b
    eor \t4\().16b, \x2\().16b, \x0\().16b
    g4_mul \r1, \r0, \dl, \dh, \t3, \t4, \t1, \t2
.endm

.macro bs_mux res, mask, a, b
    mov \res\().16b, \mask\().16b
    bsl \res\().16b, \a\().16b, \b\().16b
.endm

// ==========================================
// MACROS CHO MIXWORD
// ==========================================
.macro neon_xtime o0, o1, o2, o3, i0, i1, i2, i3, hb, tmp
    dup \hb\().2d, \i3\().d[1]
    ext \o0\().16b, \hb\().16b, \i0\().16b, #8
    ext \o1\().16b, \i0\().16b, \i1\().16b, #8
    ext \o2\().16b, \i1\().16b, \i2\().16b, #8
    ext \o3\().16b, \i2\().16b, \i3\().16b, #8
    movi \tmp\().2d, #0
    ins \tmp\().d[1], \hb\().d[0]
    eor \o0\().16b, \o0\().16b, \tmp\().16b
    eor \o1\().16b, \o1\().16b, \tmp\().16b
    eor \o2\().16b, \o2\().16b, \tmp\().16b
.endm

// =========================================================
// PACK / UNPACK (Xử lý đồng thời 8 khối 128-bit = 128 Bytes)
// =========================================================
    .global pack_neon
    .type pack_neon, %function
pack_neon:
    ld1 {v0.16b-v3.16b}, [x0], #64
    ld1 {v4.16b-v7.16b}, [x0]
    transpose_8x8_bytes_128 v0, v1, v2, v3, v4, v5, v6, v7
    movi v24.16b, #0x55
    movi v25.16b, #0x33
    movi v26.16b, #0x0F
    bfly_step v0, v1, v24, 1, v28, v29
    bfly_step v2, v3, v24, 1, v28, v29
    bfly_step v4, v5, v24, 1, v28, v29
    bfly_step v6, v7, v24, 1, v28, v29
    bfly_step v0, v2, v25, 2, v28, v29
    bfly_step v1, v3, v25, 2, v28, v29
    bfly_step v4, v6, v25, 2, v28, v29
    bfly_step v5, v7, v25, 2, v28, v29
    bfly_step v0, v4, v26, 4, v28, v29
    bfly_step v1, v5, v26, 4, v28, v29
    bfly_step v2, v6, v26, 4, v28, v29
    bfly_step v3, v7, v26, 4, v28, v29
    st1 {v0.16b-v3.16b}, [x1], #64
    st1 {v4.16b-v7.16b}, [x1]
    ret

    .global unpack_neon
    .type unpack_neon, %function
unpack_neon:
    ld1 {v0.16b-v3.16b}, [x0], #64
    ld1 {v4.16b-v7.16b}, [x0]
    movi v24.16b, #0x55
    movi v25.16b, #0x33
    movi v26.16b, #0x0F
    bfly_step v0, v1, v24, 1, v28, v29
    bfly_step v2, v3, v24, 1, v28, v29
    bfly_step v4, v5, v24, 1, v28, v29
    bfly_step v6, v7, v24, 1, v28, v29
    bfly_step v0, v2, v25, 2, v28, v29
    bfly_step v1, v3, v25, 2, v28, v29
    bfly_step v4, v6, v25, 2, v28, v29
    bfly_step v5, v7, v25, 2, v28, v29
    bfly_step v0, v4, v26, 4, v28, v29
    bfly_step v1, v5, v26, 4, v28, v29
    bfly_step v2, v6, v26, 4, v28, v29
    bfly_step v3, v7, v26, 4, v28, v29
    transpose_8x8_bytes_128 v0, v1, v2, v3, v4, v5, v6, v7
    st1 {v0.16b-v3.16b}, [x1], #64
    st1 {v4.16b-v7.16b}, [x1]
    ret

// =========================================================
// S-BOX BITSCLICED NEON (8 Blocks Parallel Execution)
// =========================================================
    .global sbox_bs_neon
    .type sbox_bs_neon, %function
sbox_bs_neon:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ld1 {v0.16b-v3.16b}, [x0]
    add x8, x0, #64
    ld1 {v4.16b-v7.16b}, [x8]

    map_M v8, v9, v10, v11, v3, v2, v1, v0
    map_M v12, v13, v14, v15, v7, v6, v5, v4

    gf16_sqr v16, v17, v18, v19, v12, v13, v14, v15, v20
    gf16_sqr v20, v21, v22, v23, v16, v17, v18, v19, v24
    gf16_inv v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29
    gf16_mul v20, v21, v22, v23, v16, v17, v18, v19, v8, v9, v10, v11, v24, v25, v26, v27, v28, v29, v30, v31
    gf16_inv v24, v25, v26, v27, v12, v13, v14, v15, v28, v29, v30, v31, v16, v17

    orr v28.16b, v8.16b, v9.16b
    orr v28.16b, v28.16b, v10.16b
    orr v28.16b, v28.16b, v11.16b

    bs_mux v16, v28, v20, v24
    bs_mux v17, v28, v21, v25
    bs_mux v18, v28, v22, v26
    bs_mux v19, v28, v23, v27

    gf16_mul v20, v21, v22, v23, v8, v9, v10, v11, v16, v17, v18, v19, v24, v25, v26, v27, v28, v29, v30, v31
    gf16_sqr v24, v25, v26, v27, v20, v21, v22, v23, v28
    gf16_inv v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v12, v13
    gf16_inv v24, v25, v26, v27, v8, v9, v10, v11, v28, v29, v30, v31, v12, v13

    orr v28.16b, v16.16b, v17.16b
    orr v28.16b, v28.16b, v18.16b
    orr v28.16b, v28.16b, v19.16b

    bs_mux v8, v28, v20, v24
    bs_mux v9, v28, v21, v25
    bs_mux v10, v28, v22, v26
    bs_mux v11, v28, v23, v27

    map_M_inv v7, v6, v5, v4, v8, v9, v10, v11
    map_M_inv v3, v2, v1, v0, v16, v17, v18, v19

    mvn v0.16b, v0.16b

    st1 {v0.16b-v3.16b}, [x0]
    st1 {v4.16b-v7.16b}, [x8]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret

// =========================================================
// INV S-BOX BITSCLICED NEON (8 Blocks Parallel Execution)
// =========================================================
    .global inv_sbox_bs_neon
    .type inv_sbox_bs_neon, %function
inv_sbox_bs_neon:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ld1 {v0.16b-v3.16b}, [x0]
    add x8, x0, #64
    ld1 {v4.16b-v7.16b}, [x8]

    mvn v0.16b, v0.16b
    map_M v16, v17, v18, v19, v3, v2, v1, v0
    map_M v8, v9, v10, v11, v7, v6, v5, v4

    gf16_inv v20, v21, v22, v23, v16, v17, v18, v19, v24, v25, v26, v27, v28, v29
    gf16_inv v0, v1, v2, v3, v8, v9, v10, v11, v24, v25, v26, v27, v28, v29

    gf16_sqr v24, v25, v26, v27, v0, v1, v2, v3, v28
    gf16_sqr v28, v29, v30, v31, v24, v25, v26, v27, v12
    gf16_sqr v24, v25, v26, v27, v28, v29, v30, v31, v12

    gf16_mul v28, v29, v30, v31, v24, v25, v26, v27, v20, v21, v22, v23, v12, v13, v14, v15, v4, v5, v6, v7

    orr v12.16b, v16.16b, v17.16b
    orr v12.16b, v12.16b, v18.16b
    orr v12.16b, v12.16b, v19.16b

    bs_mux v4, v12, v28, v0
    bs_mux v5, v12, v29, v1
    bs_mux v6, v12, v30, v2
    bs_mux v7, v12, v31, v3

    gf16_inv v24, v25, v26, v27, v4, v5, v6, v7, v28, v29, v30, v31, v12, v13
    gf16_mul v28, v29, v30, v31, v16, v17, v18, v19, v24, v25, v26, v27, v12, v13, v14, v15, v0, v1, v2, v3
    gf16_inv v24, v25, v26, v27, v28, v29, v30, v31, v12, v13, v14, v15, v0, v1

    gf16_sqr v28, v29, v30, v31, v24, v25, v26, v27, v12
    gf16_sqr v24, v25, v26, v27, v28, v29, v30, v31, v12

    orr v12.16b, v4.16b, v5.16b
    orr v12.16b, v12.16b, v6.16b
    orr v12.16b, v12.16b, v7.16b

    bs_mux v0, v12, v24, v20
    bs_mux v1, v12, v25, v21
    bs_mux v2, v12, v26, v22
    bs_mux v3, v12, v27, v23

    map_M_inv v11, v10, v9, v8, v0, v1, v2, v3
    map_M_inv v15, v14, v13, v12, v4, v5, v6, v7

    st1 {v12.16b-v15.16b}, [x0]
    st1 {v8.16b-v11.16b}, [x8]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret

// =========================================================
// MIXWORD NEON (Layout-Aware Dual-Group 128-bit Matrix Rotation)
// =========================================================
    .global mixword_neon
    .type mixword_neon, %function
mixword_neon:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    // Tải 128 bytes dữ liệu (8 khối bitslices rộng 128-bit)
    ld1 {v0.16b-v3.16b}, [x0]
    add x8, x0, #64
    ld1 {v4.16b-v7.16b}, [x8]

    // Transpose phân tách thành Nhóm 1 (Khối 0-3) và Nhóm 2 (Khối 4-7)
    trn1 v16.2d, v0.2d, v1.2d  // Nhóm 1, thanh ghi 0
    trn2 v17.2d, v0.2d, v1.2d  // Nhóm 2, thanh ghi 0
    trn1 v18.2d, v2.2d, v3.2d  // Nhóm 1, thanh ghi 1
    trn2 v19.2d, v2.2d, v3.2d  // Nhóm 2, thanh ghi 1
    trn1 v20.2d, v4.2d, v5.2d  // Nhóm 1, thanh ghi 2
    trn2 v21.2d, v4.2d, v5.2d  // Nhóm 2, thanh ghi 2
    trn1 v22.2d, v6.2d, v7.2d  // Nhóm 1, thanh ghi 3
    trn2 v23.2d, v6.2d, v7.2d  // Nhóm 2, thanh ghi 3

    // Cấp phát Stack lưu trữ kết quả trung gian để tránh tràn thanh ghi
    sub sp, sp, #128

    // SAO LƯU NHÓM 2 VÀO STACK ĐỂ TRÁNH BỊ GHI ĐÈ TRONG KHI XỬ LÝ NHÓM 1
    str q17, [sp, #64]
    str q19, [sp, #80]
    str q21, [sp, #96]
    str q23, [sp, #112]

    // --- XỬ LÝ NHÓM 1 (Khối 0-3) ---
    mov v0.16b, v16.16b
    mov v1.16b, v18.16b
    mov v2.16b, v20.16b
    mov v3.16b, v22.16b

    mov x9, #0x1111111111111111
    dup v4.2d, x9

    ushr v5.2d, v0.2d, #1
    ushr v6.2d, v1.2d, #1
    ushr v7.2d, v2.2d, #1
    ushr v8.2d, v3.2d, #1
    ushr v9.2d, v0.2d, #3
    ushr v10.2d, v1.2d, #3
    ushr v11.2d, v2.2d, #3
    ushr v12.2d, v3.2d, #3

    eor v5.16b, v5.16b, v9.16b
    eor v6.16b, v6.16b, v10.16b
    eor v7.16b, v7.16b, v11.16b
    eor v8.16b, v8.16b, v12.16b

    and v5.16b, v5.16b, v4.16b
    and v6.16b, v6.16b, v4.16b
    and v7.16b, v7.16b, v4.16b
    and v8.16b, v8.16b, v4.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    ushr v5.2d, v0.2d, #2
    ushr v6.2d, v1.2d, #2
    ushr v7.2d, v2.2d, #2
    ushr v8.2d, v3.2d, #2

    eor v13.16b, v0.16b, v5.16b
    eor v14.16b, v1.16b, v6.16b
    eor v15.16b, v2.16b, v7.16b
    eor v16.16b, v3.16b, v8.16b

    ushr v25.2d, v0.2d, #3
    ushr v26.2d, v1.2d, #3
    ushr v27.2d, v2.2d, #3
    ushr v28.2d, v3.2d, #3

    eor v13.16b, v13.16b, v25.16b
    eor v14.16b, v14.16b, v26.16b
    eor v15.16b, v15.16b, v27.16b
    eor v16.16b, v16.16b, v28.16b

    and v13.16b, v13.16b, v4.16b
    and v14.16b, v14.16b, v4.16b
    and v15.16b, v15.16b, v4.16b
    and v16.16b, v16.16b, v4.16b

    eor v13.16b, v13.16b, v9.16b
    eor v14.16b, v14.16b, v10.16b
    eor v15.16b, v15.16b, v11.16b
    eor v16.16b, v16.16b, v12.16b

    and v5.16b, v5.16b, v4.16b
    and v6.16b, v6.16b, v4.16b
    and v7.16b, v7.16b, v4.16b
    and v8.16b, v8.16b, v4.16b

    eor v5.16b, v5.16b, v13.16b
    eor v6.16b, v6.16b, v14.16b
    eor v7.16b, v7.16b, v15.16b
    eor v8.16b, v8.16b, v16.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    ushr v17.2d, v0.2d, #1
    ushr v18.2d, v1.2d, #1
    ushr v19.2d, v2.2d, #1
    ushr v20.2d, v3.2d, #1

    and v17.16b, v17.16b, v4.16b
    and v18.16b, v18.16b, v4.16b
    and v19.16b, v19.16b, v4.16b
    and v20.16b, v20.16b, v4.16b

    and v25.16b, v25.16b, v4.16b
    and v26.16b, v26.16b, v4.16b
    and v27.16b, v27.16b, v4.16b
    and v28.16b, v28.16b, v4.16b

    eor v17.16b, v17.16b, v25.16b
    eor v18.16b, v18.16b, v26.16b
    eor v19.16b, v19.16b, v27.16b
    eor v20.16b, v20.16b, v28.16b

    eor v17.16b, v17.16b, v13.16b
    eor v18.16b, v18.16b, v14.16b
    eor v19.16b, v19.16b, v15.16b
    eor v20.16b, v20.16b, v16.16b

    eor v17.16b, v17.16b, v9.16b
    eor v18.16b, v18.16b, v10.16b
    eor v19.16b, v19.16b, v11.16b
    eor v20.16b, v20.16b, v12.16b

    eor v5.16b, v25.16b, v17.16b
    eor v6.16b, v26.16b, v18.16b
    eor v7.16b, v27.16b, v19.16b
    eor v8.16b, v28.16b, v20.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    ushr v21.2d, v0.2d, #2
    ushr v22.2d, v1.2d, #2
    ushr v23.2d, v2.2d, #2
    ushr v24.2d, v3.2d, #2

    and v21.16b, v21.16b, v4.16b
    and v22.16b, v22.16b, v4.16b
    and v23.16b, v23.16b, v4.16b
    and v24.16b, v24.16b, v4.16b

    eor v21.16b, v21.16b, v13.16b
    eor v22.16b, v22.16b, v14.16b
    eor v23.16b, v23.16b, v15.16b
    eor v24.16b, v24.16b, v16.16b

    eor v21.16b, v21.16b, v17.16b
    eor v22.16b, v22.16b, v18.16b
    eor v23.16b, v23.16b, v19.16b
    eor v24.16b, v24.16b, v20.16b

    eor v21.16b, v21.16b, v9.16b
    eor v22.16b, v22.16b, v10.16b
    eor v23.16b, v23.16b, v11.16b
    eor v24.16b, v24.16b, v12.16b

    eor v5.16b, v13.16b, v21.16b
    eor v6.16b, v14.16b, v22.16b
    eor v7.16b, v15.16b, v23.16b
    eor v8.16b, v16.16b, v24.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    eor v25.16b, v25.16b, v17.16b
    eor v26.16b, v26.16b, v18.16b
    eor v27.16b, v27.16b, v19.16b
    eor v28.16b, v28.16b, v20.16b

    eor v25.16b, v25.16b, v21.16b
    eor v26.16b, v26.16b, v22.16b
    eor v27.16b, v27.16b, v23.16b
    eor v28.16b, v28.16b, v24.16b

    eor v25.16b, v25.16b, v9.16b
    eor v26.16b, v26.16b, v10.16b
    eor v27.16b, v27.16b, v11.16b
    eor v28.16b, v28.16b, v12.16b

    shl v17.2d, v17.2d, #1
    shl v18.2d, v18.2d, #1
    shl v19.2d, v19.2d, #1
    shl v20.2d, v20.2d, #1

    shl v21.2d, v21.2d, #2
    shl v22.2d, v22.2d, #2
    shl v23.2d, v23.2d, #2
    shl v24.2d, v24.2d, #2

    shl v25.2d, v25.2d, #3
    shl v26.2d, v26.2d, #3
    shl v27.2d, v27.2d, #3
    shl v28.2d, v28.2d, #3

    orr v13.16b, v13.16b, v17.16b
    orr v14.16b, v14.16b, v18.16b
    orr v15.16b, v15.16b, v19.16b
    orr v16.16b, v16.16b, v20.16b

    orr v13.16b, v13.16b, v21.16b
    orr v14.16b, v14.16b, v22.16b
    orr v15.16b, v15.16b, v23.16b
    orr v16.16b, v16.16b, v24.16b

    orr v13.16b, v13.16b, v25.16b
    orr v14.16b, v14.16b, v26.16b
    orr v15.16b, v15.16b, v27.16b
    orr v16.16b, v16.16b, v28.16b

    // Đẩy kết quả đầu ra Nhóm 1 vào Stack (Vị trí 0->64 bytes)
    st1 {v13.16b-v16.16b}, [sp]

    // --- XỬ LÝ NHÓM 2 (Khối 4-7) ---
    // Khôi phục an toàn dữ liệu đầu vào Nhóm 2 từ Stack vào v0-v3
    ldr q0, [sp, #64]
    ldr q1, [sp, #80]
    ldr q2, [sp, #96]
    ldr q3, [sp, #112]

    // Chạy lại khối xử lý logic MixWord trên Nhóm 2
    mov x9, #0x1111111111111111
    dup v4.2d, x9

    ushr v5.2d, v0.2d, #1
    ushr v6.2d, v1.2d, #1
    ushr v7.2d, v2.2d, #1
    ushr v8.2d, v3.2d, #1
    ushr v9.2d, v0.2d, #3
    ushr v10.2d, v1.2d, #3
    ushr v11.2d, v2.2d, #3
    ushr v12.2d, v3.2d, #3

    eor v5.16b, v5.16b, v9.16b
    eor v6.16b, v6.16b, v10.16b
    eor v7.16b, v7.16b, v11.16b
    eor v8.16b, v8.16b, v12.16b

    and v5.16b, v5.16b, v4.16b
    and v6.16b, v6.16b, v4.16b
    and v7.16b, v7.16b, v4.16b
    and v8.16b, v8.16b, v4.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    ushr v5.2d, v0.2d, #2
    ushr v6.2d, v1.2d, #2
    ushr v7.2d, v2.2d, #2
    ushr v8.2d, v3.2d, #2

    eor v13.16b, v0.16b, v5.16b
    eor v14.16b, v1.16b, v6.16b
    eor v15.16b, v2.16b, v7.16b
    eor v16.16b, v3.16b, v8.16b

    ushr v25.2d, v0.2d, #3
    ushr v26.2d, v1.2d, #3
    ushr v27.2d, v2.2d, #3
    ushr v28.2d, v3.2d, #3

    eor v13.16b, v13.16b, v25.16b
    eor v14.16b, v14.16b, v26.16b
    eor v15.16b, v15.16b, v27.16b
    eor v16.16b, v16.16b, v28.16b

    and v13.16b, v13.16b, v4.16b
    and v14.16b, v14.16b, v4.16b
    and v15.16b, v15.16b, v4.16b
    and v16.16b, v16.16b, v4.16b

    eor v13.16b, v13.16b, v9.16b
    eor v14.16b, v14.16b, v10.16b
    eor v15.16b, v15.16b, v11.16b
    eor v16.16b, v16.16b, v12.16b

    and v5.16b, v5.16b, v4.16b
    and v6.16b, v6.16b, v4.16b
    and v7.16b, v7.16b, v4.16b
    and v8.16b, v8.16b, v4.16b

    eor v5.16b, v5.16b, v13.16b
    eor v6.16b, v6.16b, v14.16b
    eor v7.16b, v7.16b, v15.16b
    eor v8.16b, v8.16b, v16.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    ushr v17.2d, v0.2d, #1
    ushr v18.2d, v1.2d, #1
    ushr v19.2d, v2.2d, #1
    ushr v20.2d, v3.2d, #1

    and v17.16b, v17.16b, v4.16b
    and v18.16b, v18.16b, v4.16b
    and v19.16b, v19.16b, v4.16b
    and v20.16b, v20.16b, v4.16b

    and v25.16b, v25.16b, v4.16b
    and v26.16b, v26.16b, v4.16b
    and v27.16b, v27.16b, v4.16b
    and v28.16b, v28.16b, v4.16b

    eor v17.16b, v17.16b, v25.16b
    eor v18.16b, v18.16b, v26.16b
    eor v19.16b, v19.16b, v27.16b
    eor v20.16b, v20.16b, v28.16b

    eor v17.16b, v17.16b, v13.16b
    eor v18.16b, v18.16b, v14.16b
    eor v19.16b, v19.16b, v15.16b
    eor v20.16b, v20.16b, v16.16b

    eor v17.16b, v17.16b, v9.16b
    eor v18.16b, v18.16b, v10.16b
    eor v19.16b, v19.16b, v11.16b
    eor v20.16b, v20.16b, v12.16b

    eor v5.16b, v25.16b, v17.16b
    eor v6.16b, v26.16b, v18.16b
    eor v7.16b, v27.16b, v19.16b
    eor v8.16b, v28.16b, v20.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    ushr v21.2d, v0.2d, #2
    ushr v22.2d, v1.2d, #2
    ushr v23.2d, v2.2d, #2
    ushr v24.2d, v3.2d, #2

    and v21.16b, v21.16b, v4.16b
    and v22.16b, v22.16b, v4.16b
    and v23.16b, v23.16b, v4.16b
    and v24.16b, v24.16b, v4.16b

    eor v21.16b, v21.16b, v13.16b
    eor v22.16b, v22.16b, v14.16b
    eor v23.16b, v23.16b, v15.16b
    eor v24.16b, v24.16b, v16.16b

    eor v21.16b, v21.16b, v17.16b
    eor v22.16b, v22.16b, v18.16b
    eor v23.16b, v23.16b, v19.16b
    eor v24.16b, v24.16b, v20.16b

    eor v21.16b, v21.16b, v9.16b
    eor v22.16b, v22.16b, v10.16b
    eor v23.16b, v23.16b, v11.16b
    eor v24.16b, v24.16b, v12.16b

    eor v5.16b, v13.16b, v21.16b
    eor v6.16b, v14.16b, v22.16b
    eor v7.16b, v15.16b, v23.16b
    eor v8.16b, v16.16b, v24.16b

    neon_xtime v9, v10, v11, v12, v5, v6, v7, v8, v29, v30

    eor v25.16b, v25.16b, v17.16b
    eor v26.16b, v26.16b, v18.16b
    eor v27.16b, v27.16b, v19.16b
    eor v28.16b, v28.16b, v20.16b

    eor v25.16b, v25.16b, v21.16b
    eor v26.16b, v26.16b, v22.16b
    eor v27.16b, v27.16b, v23.16b
    eor v28.16b, v28.16b, v24.16b

    eor v25.16b, v25.16b, v9.16b
    eor v26.16b, v26.16b, v10.16b
    eor v27.16b, v27.16b, v11.16b
    eor v28.16b, v28.16b, v12.16b

    shl v17.2d, v17.2d, #1
    shl v18.2d, v18.2d, #1
    shl v19.2d, v19.2d, #1
    shl v20.2d, v20.2d, #1

    shl v21.2d, v21.2d, #2
    shl v22.2d, v22.2d, #2
    shl v23.2d, v23.2d, #2
    shl v24.2d, v24.2d, #2

    shl v25.2d, v25.2d, #3
    shl v26.2d, v26.2d, #3
    shl v27.2d, v27.2d, #3
    shl v28.2d, v28.2d, #3

    orr v13.16b, v13.16b, v17.16b
    orr v14.16b, v14.16b, v18.16b
    orr v15.16b, v15.16b, v19.16b
    orr v16.16b, v16.16b, v20.16b

    orr v13.16b, v13.16b, v21.16b
    orr v14.16b, v14.16b, v22.16b
    orr v15.16b, v15.16b, v23.16b
    orr v16.16b, v16.16b, v24.16b

    orr v13.16b, v13.16b, v25.16b
    orr v14.16b, v14.16b, v26.16b
    orr v15.16b, v15.16b, v27.16b
    orr v16.16b, v16.16b, v28.16b

    // Gán đầu ra Nhóm 2 sang v28-v31
    mov v28.16b, v13.16b
    mov v29.16b, v14.16b
    mov v30.16b, v15.16b
    mov v31.16b, v16.16b

    // Đọc ngược kết quả đầu ra Nhóm 1 từ Stack vào v24-v27
    ld1 {v24.16b-v27.16b}, [sp]
    add sp, sp, #128

    // Untranspose: Gom ngược Nhóm 1 và Nhóm 2 về cấu trúc Bitslices 128-bit
    trn1 v0.2d, v24.2d, v28.2d
    trn2 v1.2d, v24.2d, v28.2d
    trn1 v2.2d, v25.2d, v29.2d
    trn2 v3.2d, v25.2d, v29.2d
    trn1 v4.2d, v26.2d, v30.2d
    trn2 v5.2d, v26.2d, v30.2d
    trn1 v6.2d, v27.2d, v31.2d
    trn2 v7.2d, v27.2d, v31.2d

    // Lưu kết quả cuối cùng ngược lại bộ nhớ
    st1 {v0.16b-v3.16b}, [x0]
    st1 {v4.16b-v7.16b}, [x8]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret

// =========================================================
// INV MIXWORD NEON (Layout-Aware Dual-Group 128-bit Matrix Rotation)
// =========================================================
    .global inv_mixword_neon
    .type inv_mixword_neon, %function
inv_mixword_neon:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ld1 {v0.16b-v3.16b}, [x0]
    add x8, x0, #64
    ld1 {v4.16b-v7.16b}, [x8]

    trn1 v16.2d, v0.2d, v1.2d
    trn2 v17.2d, v0.2d, v1.2d
    trn1 v18.2d, v2.2d, v3.2d
    trn2 v19.2d, v2.2d, v3.2d
    trn1 v20.2d, v4.2d, v5.2d
    trn2 v21.2d, v4.2d, v5.2d
    trn1 v22.2d, v6.2d, v7.2d
    trn2 v23.2d, v6.2d, v7.2d

    sub sp, sp, #128

    // SAO LƯU NHÓM 2 VÀO STACK ĐỂ TRÁNH BỊ GHI ĐÈ TRONG KHI XỬ LÝ NHÓM 1
    str q17, [sp, #64]
    str q19, [sp, #80]
    str q21, [sp, #96]
    str q23, [sp, #112]

    // --- XỬ LÝ INV MIXWORD NHÓM 1 ---
    mov v0.16b, v16.16b
    mov v1.16b, v18.16b
    mov v2.16b, v20.16b
    mov v3.16b, v22.16b

    mov x9, #0x1111111111111111
    dup v31.2d, x9

    and v8.16b, v0.16b, v31.16b
    and v9.16b, v1.16b, v31.16b
    and v10.16b, v2.16b, v31.16b
    and v11.16b, v3.16b, v31.16b

    ushr v12.2d, v0.2d, #1
    ushr v13.2d, v1.2d, #1
    ushr v14.2d, v2.2d, #1
    ushr v15.2d, v3.2d, #1
    and v12.16b, v12.16b, v31.16b
    and v13.16b, v13.16b, v31.16b
    and v14.16b, v14.16b, v31.16b
    and v15.16b, v15.16b, v31.16b

    ushr v16.2d, v0.2d, #2
    ushr v17.2d, v1.2d, #2
    ushr v18.2d, v2.2d, #2
    ushr v19.2d, v3.2d, #2
    and v16.16b, v16.16b, v31.16b
    and v17.16b, v17.16b, v31.16b
    and v18.16b, v18.16b, v31.16b
    and v19.16b, v19.16b, v31.16b

    ushr v20.2d, v0.2d, #3
    ushr v21.2d, v1.2d, #3
    ushr v22.2d, v2.2d, #3
    ushr v23.2d, v3.2d, #3
    and v20.16b, v20.16b, v31.16b
    and v21.16b, v21.16b, v31.16b
    and v22.16b, v22.16b, v31.16b
    and v23.16b, v23.16b, v31.16b

    eor v0.16b, v8.16b, v16.16b
    eor v1.16b, v9.16b, v17.16b
    eor v2.16b, v10.16b, v18.16b
    eor v3.16b, v11.16b, v19.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v20.16b, v20.16b, v12.16b
    eor v21.16b, v21.16b, v13.16b
    eor v22.16b, v22.16b, v14.16b
    eor v23.16b, v23.16b, v15.16b

    eor v20.16b, v20.16b, v16.16b
    eor v21.16b, v21.16b, v17.16b
    eor v22.16b, v22.16b, v18.16b
    eor v23.16b, v23.16b, v19.16b

    eor v20.16b, v20.16b, v4.16b
    eor v21.16b, v21.16b, v5.16b
    eor v22.16b, v22.16b, v6.16b
    eor v23.16b, v23.16b, v7.16b

    eor v0.16b, v12.16b, v20.16b
    eor v1.16b, v13.16b, v21.16b
    eor v2.16b, v14.16b, v22.16b
    eor v3.16b, v15.16b, v23.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v16.16b, v16.16b, v8.16b
    eor v17.16b, v17.16b, v9.16b
    eor v18.16b, v18.16b, v10.16b
    eor v19.16b, v19.16b, v11.16b

    eor v16.16b, v16.16b, v12.16b
    eor v17.16b, v17.16b, v13.16b
    eor v18.16b, v18.16b, v14.16b
    eor v19.16b, v19.16b, v15.16b

    eor v16.16b, v16.16b, v4.16b
    eor v17.16b, v17.16b, v5.16b
    eor v18.16b, v18.16b, v6.16b
    eor v19.16b, v19.16b, v7.16b

    eor v0.16b, v8.16b, v16.16b
    eor v1.16b, v9.16b, v17.16b
    eor v2.16b, v10.16b, v18.16b
    eor v3.16b, v11.16b, v19.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v12.16b, v12.16b, v20.16b
    eor v13.16b, v13.16b, v21.16b
    eor v14.16b, v14.16b, v22.16b
    eor v15.16b, v15.16b, v23.16b

    eor v12.16b, v12.16b, v8.16b
    eor v13.16b, v13.16b, v9.16b
    eor v14.16b, v14.16b, v10.16b
    eor v15.16b, v15.16b, v11.16b

    eor v12.16b, v12.16b, v4.16b
    eor v13.16b, v13.16b, v5.16b
    eor v14.16b, v14.16b, v6.16b
    eor v15.16b, v15.16b, v7.16b

    eor v0.16b, v12.16b, v20.16b
    eor v1.16b, v13.16b, v21.16b
    eor v2.16b, v14.16b, v22.16b
    eor v3.16b, v15.16b, v23.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v8.16b, v8.16b, v16.16b
    eor v9.16b, v9.16b, v17.16b
    eor v10.16b, v10.16b, v18.16b
    eor v11.16b, v11.16b, v19.16b

    eor v8.16b, v8.16b, v20.16b
    eor v9.16b, v9.16b, v21.16b
    eor v10.16b, v10.16b, v22.16b
    eor v11.16b, v11.16b, v23.16b

    eor v8.16b, v8.16b, v4.16b
    eor v9.16b, v9.16b, v5.16b
    eor v10.16b, v10.16b, v6.16b
    eor v11.16b, v11.16b, v7.16b

    shl v12.2d, v12.2d, #1
    shl v13.2d, v13.2d, #1
    shl v14.2d, v14.2d, #1
    shl v15.2d, v15.2d, #1

    shl v16.2d, v16.2d, #2
    shl v17.2d, v17.2d, #2
    shl v18.2d, v18.2d, #2
    shl v19.2d, v19.2d, #2

    shl v20.2d, v20.2d, #3
    shl v21.2d, v21.2d, #3
    shl v22.2d, v22.2d, #3
    shl v23.2d, v23.2d, #3

    orr v8.16b, v8.16b, v12.16b
    orr v9.16b, v9.16b, v13.16b
    orr v10.16b, v10.16b, v14.16b
    orr v11.16b, v11.16b, v15.16b

    orr v8.16b, v8.16b, v16.16b
    orr v9.16b, v9.16b, v17.16b
    orr v10.16b, v10.16b, v18.16b
    orr v11.16b, v11.16b, v19.16b

    orr v8.16b, v8.16b, v20.16b
    orr v9.16b, v9.16b, v21.16b
    orr v10.16b, v10.16b, v22.16b
    orr v11.16b, v11.16b, v23.16b

    st1 {v8.16b-v11.16b}, [sp]

    // --- XỬ LÝ INV MIXWORD NHÓM 2 ---
    // Khôi phục dữ liệu gốc Nhóm 2 từ Stack vào v0-v3
    ldr q0, [sp, #64]
    ldr q1, [sp, #80]
    ldr q2, [sp, #96]
    ldr q3, [sp, #112]

    mov x9, #0x1111111111111111
    dup v31.2d, x9

    and v8.16b, v0.16b, v31.16b
    and v9.16b, v1.16b, v31.16b
    and v10.16b, v2.16b, v31.16b
    and v11.16b, v3.16b, v31.16b

    ushr v12.2d, v0.2d, #1
    ushr v13.2d, v1.2d, #1
    ushr v14.2d, v2.2d, #1
    ushr v15.2d, v3.2d, #1
    and v12.16b, v12.16b, v31.16b
    and v13.16b, v13.16b, v31.16b
    and v14.16b, v14.16b, v31.16b
    and v15.16b, v15.16b, v31.16b

    ushr v16.2d, v0.2d, #2
    ushr v17.2d, v1.2d, #2
    ushr v18.2d, v2.2d, #2
    ushr v19.2d, v3.2d, #2
    and v16.16b, v16.16b, v31.16b
    and v17.16b, v17.16b, v31.16b
    and v18.16b, v18.16b, v31.16b
    and v19.16b, v19.16b, v31.16b

    ushr v20.2d, v0.2d, #3
    ushr v21.2d, v1.2d, #3
    ushr v22.2d, v2.2d, #3
    ushr v23.2d, v3.2d, #3
    and v20.16b, v20.16b, v31.16b
    and v21.16b, v21.16b, v31.16b
    and v22.16b, v22.16b, v31.16b
    and v23.16b, v23.16b, v31.16b

    eor v0.16b, v8.16b, v16.16b
    eor v1.16b, v9.16b, v17.16b
    eor v2.16b, v10.16b, v18.16b
    eor v3.16b, v11.16b, v19.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v20.16b, v20.16b, v12.16b
    eor v21.16b, v21.16b, v13.16b
    eor v22.16b, v22.16b, v14.16b
    eor v23.16b, v23.16b, v15.16b

    eor v20.16b, v20.16b, v16.16b
    eor v21.16b, v21.16b, v17.16b
    eor v22.16b, v22.16b, v18.16b
    eor v23.16b, v23.16b, v19.16b

    eor v20.16b, v20.16b, v4.16b
    eor v21.16b, v21.16b, v5.16b
    eor v22.16b, v22.16b, v6.16b
    eor v23.16b, v23.16b, v7.16b

    eor v0.16b, v12.16b, v20.16b
    eor v1.16b, v13.16b, v21.16b
    eor v2.16b, v14.16b, v22.16b
    eor v3.16b, v15.16b, v23.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v16.16b, v16.16b, v8.16b
    eor v17.16b, v17.16b, v9.16b
    eor v18.16b, v18.16b, v10.16b
    eor v19.16b, v19.16b, v11.16b

    eor v16.16b, v16.16b, v12.16b
    eor v17.16b, v17.16b, v13.16b
    eor v18.16b, v18.16b, v14.16b
    eor v19.16b, v19.16b, v15.16b

    eor v16.16b, v16.16b, v4.16b
    eor v17.16b, v17.16b, v5.16b
    eor v18.16b, v18.16b, v6.16b
    eor v19.16b, v19.16b, v7.16b

    eor v0.16b, v8.16b, v16.16b
    eor v1.16b, v9.16b, v17.16b
    eor v2.16b, v10.16b, v18.16b
    eor v3.16b, v11.16b, v19.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v12.16b, v12.16b, v20.16b
    eor v13.16b, v13.16b, v21.16b
    eor v14.16b, v14.16b, v22.16b
    eor v15.16b, v15.16b, v23.16b

    eor v12.16b, v12.16b, v8.16b
    eor v13.16b, v13.16b, v9.16b
    eor v14.16b, v14.16b, v10.16b
    eor v15.16b, v15.16b, v11.16b

    eor v12.16b, v12.16b, v4.16b
    eor v13.16b, v13.16b, v5.16b
    eor v14.16b, v14.16b, v6.16b
    eor v15.16b, v15.16b, v7.16b

    eor v0.16b, v12.16b, v20.16b
    eor v1.16b, v13.16b, v21.16b
    eor v2.16b, v14.16b, v22.16b
    eor v3.16b, v15.16b, v23.16b
    neon_xtime v4, v5, v6, v7, v0, v1, v2, v3, v24, v25

    eor v8.16b, v8.16b, v16.16b
    eor v9.16b, v9.16b, v17.16b
    eor v10.16b, v10.16b, v18.16b
    eor v11.16b, v11.16b, v19.16b

    eor v8.16b, v8.16b, v20.16b
    eor v9.16b, v9.16b, v21.16b
    eor v10.16b, v10.16b, v22.16b
    eor v11.16b, v11.16b, v23.16b

    eor v8.16b, v8.16b, v4.16b
    eor v9.16b, v9.16b, v5.16b
    eor v10.16b, v10.16b, v6.16b
    eor v11.16b, v11.16b, v7.16b

    shl v12.2d, v12.2d, #1
    shl v13.2d, v13.2d, #1
    shl v14.2d, v14.2d, #1
    shl v15.2d, v15.2d, #1

    shl v16.2d, v16.2d, #2
    shl v17.2d, v17.2d, #2
    shl v18.2d, v18.2d, #2
    shl v19.2d, v19.2d, #2

    shl v20.2d, v20.2d, #3
    shl v21.2d, v21.2d, #3
    shl v22.2d, v22.2d, #3
    shl v23.2d, v23.2d, #3

    orr v8.16b, v8.16b, v12.16b
    orr v9.16b, v9.16b, v13.16b
    orr v10.16b, v10.16b, v14.16b
    orr v11.16b, v11.16b, v15.16b

    orr v8.16b, v8.16b, v16.16b
    orr v9.16b, v9.16b, v17.16b
    orr v10.16b, v10.16b, v18.16b
    orr v11.16b, v11.16b, v19.16b

    orr v8.16b, v8.16b, v20.16b
    orr v9.16b, v9.16b, v21.16b
    orr v10.16b, v10.16b, v22.16b
    orr v11.16b, v11.16b, v23.16b

    mov v28.16b, v8.16b
    mov v29.16b, v9.16b
    mov v30.16b, v10.16b
    mov v31.16b, v11.16b

    // Đọc ngược kết quả đầu ra Nhóm 1 từ Stack vào v24-v27
    ld1 {v24.16b-v27.16b}, [sp]
    add sp, sp, #128

    trn1 v0.2d, v24.2d, v28.2d
    trn2 v1.2d, v24.2d, v28.2d
    trn1 v2.2d, v25.2d, v29.2d
    trn2 v3.2d, v25.2d, v29.2d
    trn1 v4.2d, v26.2d, v30.2d
    trn2 v5.2d, v26.2d, v30.2d
    trn1 v6.2d, v27.2d, v31.2d
    trn2 v7.2d, v27.2d, v31.2d

    st1 {v0.16b-v3.16b}, [x0]
    st1 {v4.16b-v7.16b}, [x8]

    // PHẦN VÁ KHÔI PHỤC EPILOGUE HOÀN CHỈNH
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret