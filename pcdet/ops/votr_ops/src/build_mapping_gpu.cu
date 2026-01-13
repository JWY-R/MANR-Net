#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "build_mapping_gpu.h"
#include "votr_cuda_utils.h"

__device__ int murmur_hash(int k, int hash_size) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % hash_size;
}

__device__ int hash(int k, int hash_size) {
    return k % hash_size;
}

__device__ void hash_table_insert(int &key, int &value, int &hash_size, int *xyz_to_vidx) {

    int hash_idx = hash(key, hash_size);
    int prob_cnt = 0;
    while(true) {
        int prev_key = atomicCAS(xyz_to_vidx + hash_idx*2 + 0, EMPTY_KEY, key);
        if (prev_key == EMPTY_KEY || prev_key == key) {
            xyz_to_vidx[hash_idx*2 + 1] = value;
            break;
        }

        hash_idx = (hash_idx + 1) % hash_size;

        prob_cnt += 1;
        if (prob_cnt >= hash_size) break;
    }
}

__global__ void downsample_with_tensor_kernel(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int num_ds_voxels,
                                                const int *v_indices, int *ds_v_indices, int *xyz_to_vidx, int *vcount) {

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    int ds_z_idx = z_idx / z_stride;
    int ds_y_idx = y_idx / y_stride;
    int ds_x_idx = x_idx / x_stride;

    if (ds_x_idx >= x_max || ds_x_idx < 0 || ds_y_idx < 0 || ds_y_idx >= y_max || ds_z_idx < 0 || ds_z_idx >= z_max) return;

    xyz_to_vidx += bs_idx * x_max * y_max * z_max;
    ds_v_indices += bs_idx * num_ds_voxels * 3;

    int ret_v = atomicExch(xyz_to_vidx + ds_x_idx * y_max * z_max + ds_y_idx * z_max + ds_z_idx, BLK_SIGNAL);
    if (ret_v == BLK_SIGNAL){
        return;
    } else if (ret_v != EMPTY_KEY) {
        ret_v = atomicExch(xyz_to_vidx + ds_x_idx * y_max * z_max + ds_y_idx * z_max + ds_z_idx, ret_v);
        return;
    } else if (ret_v == EMPTY_KEY) {
        int v_idx = atomicAdd(vcount + bs_idx, 1);
        ds_v_indices[v_idx * 3 + 0] = ds_z_idx;
        ds_v_indices[v_idx * 3 + 1] = ds_y_idx;
        ds_v_indices[v_idx * 3 + 2] = ds_x_idx;
        ret_v = atomicExch(xyz_to_vidx + ds_x_idx * y_max * z_max + ds_y_idx * z_max + ds_z_idx, v_idx);
        return;
    }

}

void downsample_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int num_ds_voxels,
                                                const int *v_indices, int *ds_v_indices, int *xyz_to_vidx, int *vcount) {

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    downsample_with_tensor_kernel<<<blocks, threads>>>(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                num_voxels, num_ds_voxels,
                                                v_indices, ds_v_indices, xyz_to_vidx, vcount);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

}

__global__ void downsample_with_hash_kernel(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int num_ds_voxels, int hash_size,
                                                const int *v_indices, int *ds_v_indices, int *xyz_to_vidx, int *vcount) {

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    int ds_z_idx = z_idx / z_stride;
    int ds_y_idx = y_idx / y_stride;
    int ds_x_idx = x_idx / x_stride;

    if (ds_x_idx >= x_max || ds_x_idx < 0 || ds_y_idx < 0 || ds_y_idx >= y_max || ds_z_idx < 0 || ds_z_idx >= z_max) return;

    xyz_to_vidx += bs_idx * hash_size * 2;
    ds_v_indices += bs_idx * num_ds_voxels * 3;

    int key = ds_x_idx * y_max * z_max + ds_y_idx * z_max + ds_z_idx;
    int hash_idx = hash(key, hash_size);
    int prob_cnt = 0;
    while(true) {
        int prev_key = atomicCAS(xyz_to_vidx + hash_idx*2 + 0, EMPTY_KEY, key);
        if (prev_key == EMPTY_KEY) {
            int v_idx = atomicAdd(vcount + bs_idx, 1);
            ds_v_indices[v_idx * 3 + 0] = ds_z_idx;
            ds_v_indices[v_idx * 3 + 1] = ds_y_idx;
            ds_v_indices[v_idx * 3 + 2] = ds_x_idx;
            xyz_to_vidx[hash_idx*2 + 1] = v_idx;
            break;
        } else if (prev_key == key) {
            break;
        }

        hash_idx = (hash_idx + 1) % hash_size;
        prob_cnt += 1;
        if (prob_cnt >= hash_size) break;
    }
}

void downsample_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int num_ds_voxels, int hash_size,
                                                const int *v_indices, int *ds_v_indices, int *xyz_to_vidx, int *vcount) {

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    downsample_with_hash_kernel<<<blocks, threads>>>(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                num_voxels, num_ds_voxels, hash_size,
                                                v_indices, ds_v_indices, xyz_to_vidx, vcount);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

}

__global__ void build_mapping_with_tensor_kernel(int x_max, int y_max, int z_max, int num_voxels,
                                                    const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx) {

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;

    int v_sum = 0;
    int bs_cnt = bs_idx - 1;
    while(bs_cnt >= 0){
        v_sum += v_bs_cnt[bs_cnt];
        bs_cnt--;
    }
    int v_idx = th_idx - v_sum;

    xyz_to_vidx[bs_idx * x_max * y_max * z_max + x_idx * y_max * z_max + y_idx * z_max + z_idx] = v_idx;
}

void build_mapping_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    build_mapping_with_tensor_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels, v_indices, v_bs_cnt, xyz_to_vidx);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void build_mapping_with_hash_kernel(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx) {

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;
    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    int v_sum = 0;
    int bs_cnt = bs_idx - 1;
    while(bs_cnt >= 0){
        v_sum += v_bs_cnt[bs_cnt];
        bs_cnt--;
    }
    int v_idx = th_idx - v_sum;

    xyz_to_vidx += bs_idx * hash_size * 2;
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return; // out of bound

    int key = x_idx * y_max * z_max + y_idx * z_max + z_idx;
    hash_table_insert(key, v_idx, hash_size, xyz_to_vidx);

    return;
}

void build_mapping_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    build_mapping_with_hash_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels, hash_size,
                                                            v_indices, v_bs_cnt, xyz_to_vidx);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
