__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int m, int n, int k)
{
	int row = get_global_id(0);
	int col = get_global_id(1);

	float res = 0.0;
	for(int i = 0; i < k; i++){
		res += A[row * k + i] * B[i * k + col];
	}

	C[row * k + col] = res;
}