#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_deform_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_deform_kernel_stack(int B, int M, int nsample, \
    const float *new_xyz, const float *new_xyz_r, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    new_xyz_r += pt_idx; //add
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * nsample;

    float radius = new_xyz_r[0];
    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    if (cnt == 0) idx[0] = -1;
}


void ball_query_deform_kernel_launcher_stack(int B, int M, int nsample,
    const float *new_xyz, const float *new_xyz_r, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){

    cudaError_t err;

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_deform_kernel_stack<<<blocks, threads>>>(B, M, nsample, new_xyz, new_xyz_r, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}