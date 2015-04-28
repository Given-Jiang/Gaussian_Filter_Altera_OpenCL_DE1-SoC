#define ROWS 576
#define COLS 720
#define CHANNELS 1


__kernel
__attribute__((task))
void Gaussian(__global uchar* restrict in_img_b, __global uchar* restrict in_img_g, __global uchar* restrict in_img_r,
		__global uchar* restrict out_img_b, __global uchar* restrict out_img_g, __global uchar* restrict out_img_r,
		const unsigned int iterations)
{
    unsigned int count = 0;

    // Filter coefficients
    float pdKernel[5][5] = {{0.0050f,0.0173f,0.0262f,0.0173f,0.0050f},
							{0.0173f,0.0598f,0.0903f,0.0598f,0.0173f},
							{0.0262f,0.0903f,0.1366f,0.0903f,0.0262f},
							{0.0173f,0.0598f,0.0903f,0.0598f,0.0173f},
							{0.0050f,0.0173f,0.0262f,0.0173f,0.0050f}};

	uchar rows_b[4 * COLS + 5];
	uchar rows_g[4 * COLS + 5];
	uchar rows_r[4 * COLS + 5];

    while(count != iterations)
    {
		#pragma unroll
		for (int i = COLS * 4 + 4; i > 0; --i)
		{
			rows_b[i] = rows_b[i - 1];
		}
		rows_b[0] = in_img_b[count];

		#pragma unroll
		for (int i = COLS * 4 + 4; i > 0; --i)
		{
			rows_g[i] = rows_g[i - 1];
		}
		rows_g[0] = in_img_g[count];

		#pragma unroll
		for (int i = COLS * 4 + 4; i > 0; --i)
		{
			rows_r[i] = rows_r[i - 1];
		}
		rows_r[0] = in_img_r[count];

		float dir_b = 0;
		float dir_g = 0;
		float dir_r = 0;
	
		#pragma unroll
		for (int i = 0; i < 5; ++i)
		{
				#pragma unroll
			for (int j = 0; j < 5; ++j)
			{
				uchar pixel_b = rows_b[i * COLS + j];
				dir_b += pixel_b * pdKernel[i][j];
				}
		}
		#pragma unroll
		for (int i = 0; i < 5; ++i)
		{
			#pragma unroll
			for (int j = 0; j < 5; ++j)
			{
				uchar pixel_g = rows_g[i * COLS + j];
				dir_g += pixel_g * pdKernel[i][j];
			}
		}
		#pragma unroll
		for (int i = 0; i < 5; ++i)
		{
			#pragma unroll
			for (int j = 0; j < 5; ++j)
			{
				uchar pixel_r = rows_r[i * COLS + j];
				dir_r += pixel_r * pdKernel[i][j];
			}
		}
		out_img_b[count] = (uchar) dir_b;
		out_img_g[count] = (uchar) dir_g;
		out_img_r[count] = (uchar) dir_r;
		count++;
	}

}


