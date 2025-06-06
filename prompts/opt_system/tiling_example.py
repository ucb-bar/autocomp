def PROMPT():
    orig_code = """\
    uint32_t b_offset = 16 * 16 * 4 * 8 * sizeof(int8_t); 

    for (int_fast32_t y = 0; y < 8; y++) {
        uint32_t b_base_y = 64 * y; 

        // Load B matrix slice
        for (int_fast32_t zo = 0; zo < 8; zo++) {
            uint32_t b_zo_offset = 4 * 16 * zo; // Number of columns per zo iteration
            for (int_fast32_t z = 0; z < 4; z++) {
                uint32_t b_index = ((zo * 4 + z) * ((16 * 4) * 16)) / 16; // Divide number of elements by 16 since scratchpad is row-indexed
                mvin3(&B[b_zo_offset + 16 * z][b_base_y], b_offset + b_index, 16 * 4, 16);
            }
        }

        for (int_fast32_t x = 0; x < 32; x++) {
            uint32_t res = 1 << 31;
            uint32_t a_base_x = 16 * x; 

            // Load A matrix slice
            for (int_fast32_t zo = 0; zo < 8; zo++) {
                uint32_t a_index = (zo * (16 * 4) * 16) / 16;
                mvin2(&A[a_base_x][64 * zo], a_index, 16 * 4, 16);
            }

            // Computation
            for (int_fast32_t zo = 0; zo < 8; zo++) {   
                uint32_t a_index = (zo * (16 * 4) * 16) / 16;
                for (int_fast32_t z = 0; z < 4; z++) {
                    uint32_t preload_flag = (zo == 0 && z == 0) ? 0 : 0x40000000;
                    for (int_fast32_t y_in_o = 0; y_in_o < 4; y_in_o++) {
                        uint32_t preload_index = ((zo * 4 + z) * ((16 * 4) * 16) + y_in_o * (16 * 16)) / 16; // Find correct scratchpad index to load B from
                        preload(b_offset + preload_index, res + (y_in_o * (16 * 16)) / 16 | preload_flag, 16, 16, 16, 16);
                        compute_preloaded(a_index + (z * (16 * 16)) / 16, ~((uint32_t)0), 16, 16, 16, 16);
                    }
                }
            }

            // Store C matrix slice
            for (int_fast32_t y_in_o = 0; y_in_o < 4; y_in_o++) {
                mvout(&C[a_base_x][b_base_y + 16 * y_in_o], res + (y_in_o * (16 * 16)) / 16, 16, 16); // Divide number of elements by 16 since accumulator is row-indexed
            }
        }
    }"""
    new_code = """\
    uint32_t b_offset = 16 * 16 * 4 * 8 * sizeof(int8_t); 

    for (int_fast32_t y = 0; y < 2; y++) { // Reduce number of y dimension outer loop iterations
        uint32_t b_base_y = 256 * y;

        // Load larger B matrix slice
        // Tiling reduces redundant loads of B matrix, reducing data movement and increasing data reuse
        for (int_fast32_t zo = 0; zo < 8; zo++) {
            uint32_t b_zo_offset = 4 * 16 * zo; // Number of columns per zo iteration
            for (int_fast32_t z = 0; z < 4; z++) {
                for (int_fast32_t y_in = 0; y_in < 4; y_in++) {
                    uint32_t b_index = (((zo * 4 + z) * 4 + y_in) * ((16 * 4) * 16)) / 16; // Divide number of elements by 16 since scratchpad is row-indexed
                    mvin3(&B[b_zo_offset + 16 * z][b_base_y + 64 * y_in], b_offset + b_index, 16 * 4, 16);
                }
            }
        }

        for (int_fast32_t x = 0; x < 32; x++) {
            uint32_t res = 1 << 31;
            uint32_t a_base_x = 16 * x;

            // Load A matrix slice
            // Tiling reduces redundant loads of A matrix, reducing data movement and increasing data reuse
            for (int_fast32_t zo = 0; zo < 8; zo++) {
                uint32_t a_index = (zo * (16 * 4) * 16) / 16;
                mvin2(&A[a_base_x][64 * zo], a_index, 16 * 4, 16);
            }

            // Computation
            for (int_fast32_t zo = 0; zo < 8; zo++) {   
                uint32_t a_index = (zo * (16 * 4) * 16) / 16;
                for (int_fast32_t z = 0; z < 4; z++) {
                    uint32_t preload_flag = (zo == 0 && z == 0) ? 0 : 0x40000000;
                    for (int_fast32_t y_in_o = 0; y_in_o < 16; y_in_o++) { // Increase number of Y dimension inner loop iterations to increase tile size
                        uint32_t preload_index = (((zo * 4 + z) * 4) * ((16 * 4) * 16) + y_in_o * (16 * 16)) / 16; // Find correct scratchpad index to load B from
                        preload(b_offset + preload_index, res + (y_in_o * (16 * 16)) / 16 | preload_flag, 16, 16, 16, 16);
                        compute_preloaded(a_index + (z * (16 * 16)) / 16, ~((uint32_t)0), 16, 16, 16, 16);
                    }
                }
            }

            // Store C matrix slice
            for (int_fast32_t y_in_o = 0; y_in_o < 16; y_in_o++) { // Move out a larger tile in the Y dimension
                mvout(&C[a_base_x][b_base_y + 16 * y_in_o], res + (y_in_o * 16 * 16) / 16, 16, 16); // Divide number of elements by 16 since accumulator is row-indexed
            }
        }
    }"""
    prompt_text = "Here is an example of increasing scratchpad tile size for the Y dimension of a 512x512 (X x Z) matrix A and 512x512 (Z x Y) matrix B multiplication. Original code:\n" + orig_code + "\nRetiled code\n" + new_code
    return prompt_text