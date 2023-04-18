__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int m, int n, int k)
{
	int row = get_local_id(0);
	int col = get_local_id(1);
	int global_row = get_group_id(0) * 16 + row;
	int global_col = get_group_id(1) * 16 + col;

	__local float Asub[16][16];
    __local float Bsub[16][16];

	float res = 0;
	int num_of_tiles = k / 16;
	for(int i = 0; i < num_of_tiles; i++){
		int tiled_row = i * 16 + row;
		int tiled_col = i * 16 + col;

		Asub[row][col] = A[global_row * k + tiled_col];
		Bsub[row][col] = B[tiled_row * n + global_col];

		barrier(CLK_LOCAL_MEM_FENCE);

		for(int j = 0; j < 16; j++)
			res += Asub[row][j] * Bsub[j][col];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	C[global_row * n + global_col] = res;
}