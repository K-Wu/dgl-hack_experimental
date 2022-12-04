/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_impl.cu
 * \brief Binary reduce implementation on cuda.
 */
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>

#include "../binary_reduce_impl.h"
#include "../csr_interface.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename Idx>
std::string print_csr(const minigun::Csr<Idx>& csr);

template <typename Idx, typename DType>
std::string print_gdata2d(runtime::NDArray a, Idx dim1, Idx dim2);

template <typename Idx, typename DType>
void print_gdata(runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    const minigun::Csr<Idx> &csr,
    Idx el_xlen,
    Idx feat_src_xlen,
    runtime::NDArray  eid);

template <typename DType>
__device__ DType gatLeakyReluExp(DType val, DType slope) {
    return val > 0 ? exp(val) : exp(slope * val);
}

template <typename Idx, typename DType>
__global__ void gatExpLeakyReluSumKernel(GatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    //extern __shared__ DType er[];
    Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
    Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
    Idx stride_x = blockDim.x * gridDim.x;
    Idx stride_y = blockDim.y * gridDim.y;
    Idx dst_vid = ty;
    Idx e_xlen = gdata.e_xlen;
    while (dst_vid < csr.row_offsets.length) {
        Idx start_off = *(csr.row_offsets.data + dst_vid);
        Idx end_off = *(csr.row_offsets.data + dst_vid + 1);
        Idx feat_idx = tx;
        while (feat_idx < e_xlen) {
            // 1. Load dstnation vertex into shared memory
            Idx feat_off_dst = dst_vid * e_xlen + feat_idx;
            //er[threadIdx.x] = gdata.er[feat_off_dst];
            //__syncthreads();
            // 2. Do the computation
            DType sum = 0.;
            for (Idx eid=start_off; eid<end_off; ++eid) {
                Idx src_id = *(csr.column_indices.data + eid);
                Idx feat_off_src = src_id * e_xlen + feat_idx;
                //DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + er[threadIdx.x], gdata.leaky_relu_slope);
                DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst], gdata.leaky_relu_slope);
                gdata.exp[Idx(gdata.eids[eid] * e_xlen) + feat_idx] = tmp;
                sum += tmp;
            }
            gdata.sum[Idx(dst_vid*e_xlen) + feat_idx] = sum;
            feat_idx += stride_x;
        }
        dst_vid += stride_y;
    }
}

template <typename Idx, typename DType>
__global__ void gatSumProdZipDivKernel(GatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx dst_vid = blockIdx.y;
    Idx stride_vid =  gridDim.y;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx e_xlen = gdata.e_xlen;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (dst_vid < csr.row_offsets.length-1) {
        Idx start_off = *(csr.row_offsets.data + dst_vid);
        Idx end_off = *(csr.row_offsets.data + dst_vid + 1);
        Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                for (Idx eid=start_off; eid<end_off; eid++) {
                    Idx src_vid = csr.column_indices.data[eid];
                    s +=  gdata.exp[gdata.eids[eid] * e_xlen + head_idx] / gdata.sum[dst_vid*e_xlen + head_idx] 
                                        * gdata.feat_src[src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx];
                }
                gdata.ret[dst_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx] = s;
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        dst_vid += stride_vid;
    }
}

/*** Implement the logic of computing grad_feat_src.
    feat_src is of dimension: N * num_heads * num_hidden
    exp is of dimension: M * num_heads
    sum is of dimension: N * num_heads
    * means element-wise mutliplication
    In forward computation: out = sum([feat_src[e.src] * exp[e.eid]/sum[curnode] for e in curnode.inedges]),
    In backward computation: grad_feat_src[curnode] = sum([grad_out[e.dst] * exp[e.eid]/sum[e.dst] for e in curnode.outedges])
***/
template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradFeatSrc(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (src_vid < csr.row_offsets.length -1) {
        Idx start_off = csr.row_offsets.data[src_vid];
        Idx end_off = csr.row_offsets.data[src_vid+1];
        Idx head_idx = blockIdx.x * blockDim.x  + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx eid = gdata.eids[e];
                    Idx dst_id = csr.column_indices.data[e];
                    // TODO: maybe it's better to cache exp/sum to reduce mem traffic as well as redundant computation?
                    s += gdata.exp[eid*e_xlen + head_idx] / gdata.sum[dst_id*e_xlen + head_idx]
                        * gdata.grad_out[dst_id*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx];
                }
                gdata.grad_feat_src[src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx] = s;
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}
/***
Implement the logic of computing grad_el. 
Dimension of grad_out: N * num_heads * num_hidden
             grad_el:  N * num_heads
             grad_er:  N * num_heads
             el:       N * num_heads
             er:       N * num_heads
             exp:      M * num_heads
             sum:      N * num_heads
             feat_src: N * num_heads * num_hidden 

In forward computation: gdata.exp = [exp(leaky_relu(e.el[src] + e.el[dst])) for e in curnode.inedges]
                        gdata.sum[curnode] = sum([exp[e.eid] for e in curnode.inedges])
                        out[curnode] = sum([gdata.exp[e.eid] / gdata.sum[curnode] * gdata.feat_src[e.src] for e in curnode.inedges])
In backward computation:
                        grad_er = sum([grad_exp[e.eid] * exp(leaky_relu(gdata.el[src]+ gdata.er[dst])) * grad_leaky_relu(gdata.el[src] + gdata.er[dst]) for e in curnode.inedges])
                        grad_el = sum([grad_exp[e.eid] * leaky_relu(gdata.el[src] + gdata.er[dst]) * grad_leaky_relu(gdata.el[src] + gdata.er[dst]) for e in curnode.outedges])
                        grad_exp = [grad_out[e.dst] * (feat_src[e.src] - out[e.dst])/sum[e.dst] for e in outedges]
***/
template <typename DType>
__device__ DType gradLeaky(DType val, DType slope) {
    return val > 0 ? 1 : slope;
}

//12500
template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradElEr(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (src_vid < csr.row_offsets.length -1) {
        Idx start_off = csr.row_offsets.data[src_vid];
        Idx end_off = csr.row_offsets.data[src_vid+1];
        Idx head_idx = blockIdx.x * blockDim.x  + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                Idx feat_src_offset = src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                Idx src_node_feat_offset = src_vid*e_xlen + head_idx;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx edge_offset = gdata.eids[e] * e_xlen + head_idx;
                    Idx dst_vid = csr.column_indices.data[e];
                    Idx dst_node_feat_offset = dst_vid*e_xlen + head_idx;
                    Idx dst_out_offset = dst_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                    DType grad_exp = gdata.grad_out[dst_out_offset]* (gdata.feat_src[feat_src_offset]- gdata.ret[dst_out_offset])/gdata.sum[dst_node_feat_offset] ;
                    DType tmp_sum = gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
                    DType tmp2 = grad_exp * gdata.exp[edge_offset] * gradLeaky(tmp_sum, gdata.leaky_relu_slope);
                    s += tmp2;
                    atomicAdd(gdata.grad_er + dst_node_feat_offset, tmp2);
                }
                atomicAdd(gdata.grad_el + src_node_feat_offset , s);
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}

// 11685 basic version for num_hidden 32 num_heads 8
template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradElEr3(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.y;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (src_vid < csr.row_offsets.length -1) {
        Idx start_off = csr.row_offsets.data[src_vid];
        Idx end_off = csr.row_offsets.data[src_vid+1];
        Idx head_idx = threadIdx.y;
        while (head_idx < e_xlen) {
            Idx feat_idx = blockIdx.x * blockDim.x  + threadIdx.x;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                Idx feat_src_offset = src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                Idx src_node_feat_offset = src_vid*e_xlen + head_idx;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx edge_offset = gdata.eids[e] * e_xlen + head_idx;
                    Idx dst_vid = csr.column_indices.data[e];
                    Idx dst_node_feat_offset = dst_vid*e_xlen + head_idx;
                    Idx dst_out_offset = dst_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                    DType a = gdata.grad_out[dst_out_offset] * gdata.feat_src[feat_src_offset];
                    DType a1 = a/gdata.sum[dst_node_feat_offset];
                    DType b = gdata.grad_out[dst_out_offset] * gdata.ret[dst_out_offset];
                    DType b1 = b/gdata.sum[dst_node_feat_offset];
                    DType c = -1 * b1;
                    DType ee = a1 + c;
                    DType tmp_sum = gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
                    DType g = ee * gdata.exp[edge_offset];
                    DType h = tmp_sum > 0? 1 : gdata.leaky_relu_slope;
                    DType k = g * h;
                    s += k;
                    atomicAdd(gdata.grad_er + dst_node_feat_offset, k);
                }
                atomicAdd(gdata.grad_el + src_node_feat_offset , s);
                feat_idx += blockDim.x*gridDim.x;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}

// 13213: v3 with tiling for pubmed num_hidden 32 num_heads 8
template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradElEr4(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.y;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    Idx tiled_x = threadIdx.x % 8 + 8 * (threadIdx.y % 4);
    Idx tiled_y = threadIdx.x / 8 + 4 * (threadIdx.y / 4);
    while (src_vid < csr.row_offsets.length -1) {
        Idx start_off = csr.row_offsets.data[src_vid];
        Idx end_off = csr.row_offsets.data[src_vid+1];
        Idx head_idx = tiled_y;
        while (head_idx < e_xlen) {
            Idx feat_idx = blockIdx.x * blockDim.x  + threadIdx.x;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                Idx feat_src_offset = src_vid*gdata.feat_src_xlen + tiled_y*32 + tiled_x;
                Idx src_node_feat_offset = src_vid*e_xlen + head_idx;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx edge_offset = gdata.eids[e] * e_xlen + head_idx;
                    Idx dst_vid = csr.column_indices.data[e];
                    Idx dst_node_feat_offset = dst_vid*e_xlen + head_idx;
                    Idx dst_out_offset = dst_vid*gdata.feat_src_xlen + tiled_y*32 + tiled_x;
                    DType a = gdata.grad_out[dst_out_offset] * gdata.feat_src[feat_src_offset];
                    DType a1 = a/gdata.sum[dst_node_feat_offset];
                    DType b = gdata.grad_out[dst_out_offset] * gdata.ret[dst_out_offset];
                    DType b1 = b/gdata.sum[dst_node_feat_offset];
                    DType c = -1 * b1;
                    DType ee = a1 + c;
                    DType tmp_sum = gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
                    DType g = ee * gdata.exp[edge_offset];
                    DType h = tmp_sum > 0? 1 : gdata.leaky_relu_slope;
                    DType k = g * h;
                    s += k;
                    atomicAdd(gdata.grad_er + dst_node_feat_offset, k);
                }
                atomicAdd(gdata.grad_el + src_node_feat_offset , s);
                feat_idx += blockDim.x*gridDim.x;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}

// with shared memory and tiling
template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradElEr5(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    extern __shared__ float s[8*32 + 32];
    float * feat_src = s;
    float * er = s + 8*32;
    // TO be done
    // init, __synchrounize, and access
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.y;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    Idx tiled_x = threadIdx.x % 8 + 8 * (threadIdx.y % 4);
    Idx tiled_y = threadIdx.x / 8 + 4 * (threadIdx.y / 4);
    while (src_vid < csr.row_offsets.length -1) {
        Idx start_off = csr.row_offsets.data[src_vid];
        Idx end_off = csr.row_offsets.data[src_vid+1];
        Idx head_idx = tiled_y;
        while (head_idx < e_xlen) {
            Idx feat_idx = blockIdx.x * blockDim.x  + threadIdx.x;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                Idx feat_src_offset = src_vid*gdata.feat_src_xlen + tiled_y*32 + tiled_x;
                Idx src_node_feat_offset = src_vid*e_xlen + head_idx;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx edge_offset = gdata.eids[e] * e_xlen + head_idx;
                    Idx dst_vid = csr.column_indices.data[e];
                    Idx dst_node_feat_offset = dst_vid*e_xlen + head_idx;
                    Idx dst_out_offset = dst_vid*gdata.feat_src_xlen + tiled_y*32 + tiled_x;
                    DType a = gdata.grad_out[dst_out_offset] * gdata.feat_src[feat_src_offset];
                    DType a1 = a/gdata.sum[dst_node_feat_offset];
                    DType b = gdata.grad_out[dst_out_offset] * gdata.ret[dst_out_offset];
                    DType b1 = b/gdata.sum[dst_node_feat_offset];
                    DType c = -1 * b1;
                    DType ee = a1 + c;
                    DType tmp_sum = gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
                    DType g = ee * gdata.exp[edge_offset];
                    DType h = tmp_sum > 0? 1 : gdata.leaky_relu_slope;
                    DType k = g * h;
                    s += k;
                    atomicAdd(gdata.grad_er + dst_node_feat_offset, k);
                }
                atomicAdd(gdata.grad_el + src_node_feat_offset , s);
                feat_idx += blockDim.x*gridDim.x;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}

void FusedGatKernelImpl(
    const CSRWrapper& graph,
    runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    float slope) {
        typedef int32_t Idx;
        typedef float DType;
        const Idx MAX_NBLKS = 65535;
        const Idx MAX_NTHRS = 1024;
        // zero out ret, and packing feat_src, el, er, ret, graph together into one struct using raw float pointers
        // get csr matrix
        GatFusedData<Idx, DType> gdata;
        int64_t el_xlen =  utils::ComputeXLength(el);
        int64_t feat_src_xlen =  utils::ComputeXLength(feat_src);
        int64_t ret_len =  utils::ComputeXLength(ret);
        gdata.feat_src = static_cast<DType*>(feat_src->data);
        gdata.el = static_cast<DType*>(el->data);
        gdata.er = static_cast<DType*>(er->data);
        gdata.sum = static_cast<DType*>(sum->data);
        gdata.exp = static_cast<DType*>(exp->data);
        gdata.ret = static_cast<DType*>(ret->data);
        gdata.leaky_relu_slope = slope;
        gdata.n = el.GetSize()/sizeof(DType)/el_xlen; 
        gdata.e_xlen = el_xlen;
        gdata.feat_src_xlen =  feat_src_xlen;
        gdata.feat_src_hidden = feat_src_xlen/el_xlen;
        gdata.ret_xlen = ret_len;
        auto incsr = graph.GetInCSRMatrix();
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
        gdata.eids = static_cast<Idx*>(incsr.data->data);
        // print eids
        // printf("eids: ");
        // for (Idx i=0; i<gdata.n; ++i) {
        //     for (Idx j=0; j<el_xlen; ++j) {
        //         printf("%d ", gdata.eids[i*el_xlen + j]);
        //     }
        //     printf("\n");
        // }
        // write a device function and call it from here
        //LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" << feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen 
        //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  << " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
        //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph csr column indices length:" << csr.column_indices.length;

        // Configure kernel launch parameters.
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        int nthrs_x = 32;
        int nthrs_y = 1;
        int nblks_x = (el_xlen + nthrs_x-1)/(nthrs_x);
        int nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nblks(nblks_x, nblks_y);
        const dim3 nthrs(nthrs_x, nthrs_y);
        //LOG(INFO) << "kernel1 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;

        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,ret,csr,el_xlen, feat_src_xlen, incsr.data);
        //gatExpLeakyReluSumKernel<<<nblks, nthrs, el_xlen*sizeof(DType), thr_entry->stream>>>(gdata, csr);
        gatExpLeakyReluSumKernel<<<nblks, nthrs, 0, thr_entry->stream>>>(gdata, csr);
        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,ret,csr,el_xlen, feat_src_xlen);
        nthrs_x = utils::FindNumThreads(el_xlen, 64);
        nthrs_y = utils::FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS/nthrs_x);
        nblks_x = 1;
        nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nthrs2(nthrs_x, nthrs_y);
        const dim3 nblks2(nblks_x, nblks_y);
        //LOG(INFO) << "kernel2 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        //    printf("n_rows: %d\n", csr.row_offsets.length-1);
        //    printf("e_xlen: %d\n", gdata.e_xlen);
        //    printf("hidden_xlen: %d\n", gdata.feat_src_xlen/gdata.e_xlen);
        //    printf("stride_head: %d\n", nblks_x * nthrs_x);
        //    printf("stride_vid: %d\n", nblks_y);
        //    printf("dst_vid: %d\n", nthrs_y);
        gatSumProdZipDivKernel<<<nblks2, nthrs2, 0, thr_entry->stream>>>(gdata, csr);
}

template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(const minigun::IntArray1D<Idx>& array, Idx eid) {
  Idx lo = 0, hi = array.length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (__ldg(array.data + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (__ldg(array.data + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename Idx, typename DType>
// No need to do special pre-processing
__global__ void LoadBalanceNbAccessKernel(minigun::Csr<Idx> csr, DType* feat, Idx feat_len, Idx num_nodes, Idx num_edges) {
    Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
    Idx stride_y = blockDim.y * gridDim.y;
    Idx eid = ty;
    while (eid < num_edges) {
        Idx src = BinarySearchSrc<Idx>(csr.row_offsets, eid);
        Idx dst = __ldg(csr.column_indices.data + eid);
        Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
        const Idx stride_x = blockDim.x * gridDim.x;
        DType* srcoff = feat + src * feat_len;
        DType* dstoff = feat + dst * feat_len;
        while (tx < feat_len) {
            DType src_feat = __ldg(srcoff + tx);
            //DType tmp_dst = __ldg(dstoff + tx);
            //for (Idx i=0; i<20; ++i) {
            //    src_feat = src_feat + i;
            //}
            tx += stride_x;
        }
        eid += stride_y;
    }
}
template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessKernel(minigun::Csr<Idx> csr, DType* feat, Idx feat_len, Idx num_nodes, Idx num_edges) {
    //extern __shared__ DType dst_feat[];
    Idx dst_id = blockIdx.x;
    Idx beg = __ldg(csr.row_offsets.data + dst_id);
    Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
    Idx tx = threadIdx.x;
    for(; tx<feat_len; tx += blockDim.x) {
        //dst_feat[tx] = __ldg(feat + dst_id * feat_len + tx);
        //__syncthreads();
        for(; beg < end; ++beg) {
            Idx src_id = __ldg(csr.column_indices.data + beg);
            DType src_feat = __ldg(feat + src_id * feat_len + tx);
            //for (Idx i=0; i<20; ++i) {
            //    src_feat = src_feat + i;
            //}
        }
    }
}
template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessKernel2(minigun::Csr<Idx> csr, DType* feat, Idx feat_len, Idx num_nodes, Idx num_edges) {
    //extern __shared__ DType dst_feat[];
    Idx dst_id = blockIdx.x;
    Idx beg = __ldg(csr.row_offsets.data + dst_id);
    Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
    Idx tx = threadIdx.x;
    for(; tx<feat_len; tx += blockDim.x) {
        //dst_feat[tx] = __ldg(feat + dst_id * feat_len + tx);
        //__syncthreads();
        for(; beg < end; ++beg) {
            //Idx src_id = __ldg(csr.column_indices.data + beg);
            //DType src_feat = __ldg(feat + src_id * feat_len + tx);
            Idx src_id = csr.column_indices.data[beg];
            DType src_feat = feat[src_id*feat_len + tx];
            //for (Idx i=0; i<20; ++i) {
            //    src_feat = src_feat + i;
            //}
        }
    }
}
template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessDegIncKernel(minigun::Csr<Idx> csr, DType* feat, Idx feat_len, Idx num_nodes, Idx num_edges, Idx* deg_inc_node_map) {
    //extern __shared__ DType dst_feat[];
    if (blockIdx.x < num_nodes) {
        //Idx dst_id = __ldg(deg_inc_node_map + blockIdx.x);
        Idx dst_id = blockIdx.x;
        Idx beg = __ldg(csr.row_offsets.data + dst_id);
        Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
        Idx tx = threadIdx.x;
        for(; tx<feat_len; tx += blockDim.x) {
            //dst_feat[tx] = __ldg(feat + dst_id * feat_len + tx);
            //__syncthreads();
            for(; beg < end; ++beg) {
                Idx src_id = __ldg(csr.column_indices.data + beg);
                DType src_feat = __ldg(feat + src_id * feat_len + tx);
                //for (Idx i=0; i<20; ++i) {
                //    src_feat = src_feat + i;
                //}
            }
        }
    }
}
template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessKernelWithAtomic(minigun::Csr<Idx> csr, DType* feat, Idx feat_len, Idx num_nodes, Idx num_edges, Idx* blk_id, Idx* sorted_node_map) {
    //extern __shared__ DType dst_feat[];
    __shared__ Idx dynamic_block_id[1];
    if (threadIdx.x == 0) {
        dynamic_block_id[0] = atomicAdd(blk_id, 1);
    }
    __syncthreads();
    Idx dst_id = blockIdx.x;
    Idx beg = __ldg(csr.row_offsets.data + dst_id);
    Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
    Idx tx = threadIdx.x;
    for(; tx<feat_len; tx += blockDim.x) {
        //dst_feat[threadIdx.x] = __ldg(feat + dst_id * feat_len + tx);
        //__syncthreads();
        for(; beg < end; ++beg) {
            Idx src_id = __ldg(csr.column_indices.data + beg);
            DType src_feat = __ldg(feat + src_id * feat_len + tx);
            //for (Idx i=0; i<20; ++i) {
            //    src_feat = src_feat + i;
            //}
        }
    }
}

template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessKernelWithDynamicBlkId(minigun::Csr<Idx> csr, DType* feat, Idx feat_len, Idx num_nodes, Idx num_edges, Idx* blk_id, Idx* sorted_node_map) {
    __shared__ Idx dynamic_block_id[1];
    if (threadIdx.x == 0) {
        dynamic_block_id[0] = atomicAdd(blk_id, 1);
    }
    __syncthreads();
    if (dynamic_block_id[0] < num_nodes) {
        //Idx dst_id = __ldg(sorted_node_map + dynamic_block_id[0]);
        Idx dst_id = dynamic_block_id[0];
        Idx beg = __ldg(csr.row_offsets.data + dst_id);
        Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
        //if (threadIdx.x == 0) {
        //    printf("dst_id:%d degree:%d\n", dst_id, end-beg);
        //}
        Idx tx = threadIdx.x;
        for(; tx<feat_len; tx += blockDim.x) {
            for(; beg < end; ++beg) {
                Idx src_id = __ldg(csr.column_indices.data + beg);
                DType src_feat = __ldg(feat + src_id * feat_len + tx);
            }
        }
    }
}

template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessKernelWithDynamicBlkIdNoAtomic(minigun::Csr<Idx> csr, DType* feat, Idx feat_len, Idx num_nodes, Idx num_edges, Idx* blk_id, Idx* sorted_node_map) {
    if (blockIdx.x < num_nodes) {
        //Idx dst_id = __ldg(sorted_node_map + blockIdx.x);
        Idx dst_id = blockIdx.x;
        Idx beg = __ldg(csr.row_offsets.data + dst_id);
        Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
        Idx tx = threadIdx.x;
        for(; tx<feat_len; tx += blockDim.x) {
            for(; beg < end; ++beg) {
                Idx src_id = __ldg(csr.column_indices.data + beg);
                DType src_feat = __ldg(feat + src_id * feat_len + tx);
            }
        }
    }
}

template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessKernelWithDynamicBlkIdNoAtomicSharding(minigun::Csr<Idx> csr,
     DType* feat,
     Idx feat_len,
     Idx num_nodes,
     Idx num_edges, 
     Idx nodes_per_blk, 
     Idx new_warp_size,
     Idx* sorted_node_map) {
    Idx new_warp_id = threadIdx.x / new_warp_size; 
    Idx tx = threadIdx.x % new_warp_size;
    Idx start_off = blockIdx.x * nodes_per_blk;
    if (start_off + new_warp_id < num_nodes) {
        //Idx dst_id = __ldg(sorted_node_map + start_off + new_warp_id);
        Idx dst_id = start_off + new_warp_id;
        Idx beg = __ldg(csr.row_offsets.data + dst_id);
        Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
        for(; tx<feat_len; tx += new_warp_size) {
            for(; beg<end; ++beg) {
                Idx src_id = __ldg(csr.column_indices.data + beg);
                DType src_feat = __ldg(feat + src_id * feat_len + tx);
            }
        }
    }
}

template<typename Idx, typename DType>
__global__ void FeatureAdaptiveNbAccessKernelWithAtomicSharding(minigun::Csr<Idx> csr,
     DType* feat,
     Idx feat_len,
     Idx num_nodes,
     Idx num_edges, 
     Idx nodes_per_blk, 
     Idx new_warp_size,
     Idx* blk_id,
     Idx* sorted_node_map) {
    __shared__ Idx dynamic_block_id[1];
    if (threadIdx.x == 0) {
        dynamic_block_id[0] = atomicAdd(blk_id, 1);
    }
    __syncthreads();
    Idx new_warp_id = threadIdx.x / new_warp_size; 
    Idx tx = threadIdx.x % new_warp_size;
    Idx start_off = dynamic_block_id[0] * nodes_per_blk;
    if (start_off + new_warp_id < num_nodes) {
        //Idx dst_id = __ldg(sorted_node_map + start_off + new_warp_id);
        Idx dst_id = start_off + new_warp_id;
        Idx beg = __ldg(csr.row_offsets.data + dst_id);
        Idx end = __ldg(csr.row_offsets.data + dst_id + 1);
        for(; tx<feat_len; tx += new_warp_size) {
            for(; beg<end; ++beg) {
                Idx src_id = __ldg(csr.column_indices.data + beg);
                DType src_feat = __ldg(feat + src_id * feat_len + tx);
            }
        }
    }
}

template<typename Idx>
int launch_lb(minigun::Csr<Idx> csr, float* feat, Idx feat_len, Idx num_nodes, Idx num_edges) {
    const int MAX_NTHREADS = 1024;
    const int MAX_NBLKS= 65535;
    const int nt = utils::FindNumThreads(feat_len, 64);
    const int ty = MAX_NTHREADS / nt;
    const dim3 nthrs(nt, ty);

    const Idx M = num_edges;
    const int by = std::min((M + ty - 1) / ty, static_cast<Idx>(MAX_NBLKS));
    const int data_num_blocks = (feat_len + (nt * 2) - 1) / (nt * 2);
    const dim3 nblks(data_num_blocks, by);

    auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
    auto beg = std::chrono::steady_clock::now();
    LoadBalanceNbAccessKernel<Idx, float>
        <<<nblks, nthrs, 0, thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    int ret = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
    return ret;
}

template<typename Idx>
int launch_fa_small_feature(minigun::Csr<Idx> csr, float* feat, Idx feat_len, Idx num_nodes, Idx num_edges, int mode, Idx* node_map) {
    const int MIN_NTHREADS = 64;
    auto beg = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
    const int nthrs = MIN_NTHREADS;
    const int new_warp_size = max(1, utils::FindNumThreads(feat_len, nthrs)); // Will be 32/16/8/4/2/1
    const int nodes_per_blk = max(nthrs/new_warp_size, 1);
    const int nblks = (num_nodes + nodes_per_blk -1)/nodes_per_blk;
    //LOG(INFO) << "Launch with nblks:" << nblks << " with nthrs:" << nthrs << " new_warp_size:" <<new_warp_size << " nodes per blk:" << nodes_per_blk;
    if (mode == 2) {
        Idx* blk_id_ptr; 
        cudaCheck(cudaMalloc(&blk_id_ptr, sizeof(Idx)));
        cudaCheck(cudaMemset(blk_id_ptr, 0, sizeof(Idx)));
        cudaDeviceSynchronize();
        beg = std::chrono::steady_clock::now();
        FeatureAdaptiveNbAccessKernelWithAtomicSharding<Idx, float>
            <<<nblks, nthrs, 0, thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges, nodes_per_blk, new_warp_size, blk_id_ptr, node_map);
    } else if (mode == 3) {
        beg = std::chrono::steady_clock::now();
        FeatureAdaptiveNbAccessKernelWithDynamicBlkIdNoAtomicSharding<Idx, float>
            <<<nblks, nthrs, 0, thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges, nodes_per_blk, new_warp_size, node_map);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    return  std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
}

template<typename Idx>
int launch_fa_large_feature(minigun::Csr<Idx> csr, float* feat, Idx feat_len, Idx num_nodes, Idx num_edges, int mode, Idx* node_map) {
    const int MAX_NTHREADS = 256;
    const int MIN_NTHREADS = 64;
    auto beg = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
    const int nblks = num_nodes;
    const int nthrs = max(1, utils::FindNumThreads(feat_len, MAX_NTHREADS));

    if (mode == 0) {
        beg = std::chrono::steady_clock::now();
        //if (node_map == nullptr) {
            FeatureAdaptiveNbAccessKernel<Idx, float>
                <<<nblks, nthrs, 0, thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges);
        //} else {
        //    FeatureAdaptiveNbAccessDegIncKernel<Idx, float>
        //        <<<nblks, nthrs, 0, thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges, node_map);
        //}
    } else if (mode == 1) {
        if (node_map == nullptr) {
            LOG(FATAL) << "Mode 1 requires node map to be non-empty";
        }
        Idx* blk_id_ptr; 
        cudaCheck(cudaMalloc(&blk_id_ptr, sizeof(Idx)));
        cudaCheck(cudaMemset(blk_id_ptr, 0, sizeof(Idx)));
        cudaDeviceSynchronize();
        const int shared_memory_size_bytes = sizeof(Idx);

        beg = std::chrono::steady_clock::now();
        FeatureAdaptiveNbAccessKernelWithAtomic<Idx, float>
            <<<nblks, nthrs, shared_memory_size_bytes , thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges, blk_id_ptr, node_map);
    } else if (mode == 2) {
        //if (node_map == nullptr) {
            //LOG(FATAL) << "Mode 2 requires node map to be non-empty";
        //}
        Idx* blk_id_ptr; 
        cudaCheck(cudaMalloc(&blk_id_ptr, sizeof(Idx)));
        cudaCheck(cudaMemset(blk_id_ptr, 0, sizeof(Idx)));
        cudaDeviceSynchronize();
        const int shared_memory_size_bytes = sizeof(Idx);

        beg = std::chrono::steady_clock::now();
        FeatureAdaptiveNbAccessKernelWithDynamicBlkId<Idx, float>
            <<<nblks, nthrs, shared_memory_size_bytes , thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges, blk_id_ptr, node_map);
    } else if (mode == 3) {
        beg = std::chrono::steady_clock::now();
        FeatureAdaptiveNbAccessKernelWithDynamicBlkIdNoAtomic<Idx, float>
            <<<nblks, nthrs, 0, thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges, nullptr, node_map);
    }
    else if (mode == 4) {
        beg = std::chrono::steady_clock::now();
        int nthrs_const = 256;
        FeatureAdaptiveNbAccessDegIncKernel<Idx, float>
            <<<nblks, nthrs, 0, thr_entry->stream>>>(csr, feat, feat_len, num_nodes, num_edges, node_map);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    return  std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
}

template<typename Idx>
int launch_fa_small(minigun::Csr<Idx> csr, float* feat, Idx feat_len, Idx num_nodes, Idx num_edges, int mode=0, Idx* node_map=nullptr) {
    int ret = 0; 
    if (mode < 2) {
        ret = launch_fa_small_feature(csr, feat, feat_len, num_nodes, num_edges, mode, node_map);
    } else {
        ret = launch_fa_small_feature(csr, feat, feat_len, num_nodes, num_edges, mode, node_map);
    }
    return ret;
}

void NbAccessImpl(
    const CSRWrapper& graph,
    runtime::NDArray feat,
    runtime::NDArray node_map,
    runtime::NDArray deg_inc_node_map){
        int32_t feat_len = utils::ComputeXLength(feat);
        float* feat_ptr = static_cast<float*>(feat->data);
        auto incsr = graph.GetInCSRMatrix();
        minigun::Csr<int32_t> csr = utils::CreateCsr<int32_t>(incsr.indptr, incsr.indices);
        int32_t num_nodes = csr.row_offsets.length;
        int32_t num_edges = csr.column_indices.length;
        LOG(INFO)<<"\nfeat_len:" << feat_len <<" num_nodes:" << num_nodes << " num_edges:" << num_edges;
        float avg = 0.;
        int times = 15;
        int warm_up_times = 5;
        for (int i=0; i<times; i++){
            auto ret =  launch_lb<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges);
            if (i >= warm_up_times) {
                avg += ret;
            }
        }
        LOG(INFO) << "On average baseline takes: " << avg/(times-warm_up_times)  << " us";
        avg = 0.;
        for (int i=0; i<times; i++){
            //int32_t* node_map_ptr = static_cast<int32_t*>(node_map->data);
            //// Reset the number of nodes
            //num_nodes = node_map.GetSize()/sizeof(int32_t);
            auto ret =  launch_fa_large_feature<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 4, nullptr);
            if (i >= warm_up_times) {
                avg += ret;
            }
        }
        LOG(INFO) << "On average basic takes: " << avg/(times-warm_up_times)  << " us";
        if (feat_len >= 64) {
            //avg = 0.;
            //for (int i=0; i<times; i++){
            //    int32_t* node_map_ptr = static_cast<int32_t*>(node_map->data);
            //    // Reset the number of nodes
            //    num_nodes = node_map.GetSize()/sizeof(int32_t);
            //    auto ret = launch_fa_large_feature<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 1, node_map_ptr);
            //    if (i >= warm_up_times) {
            //        avg += ret;
            //    }
            //}
            //LOG(INFO) << "On average FA with atomic takes: " << avg/(times-warm_up_times)  << " us";
            //avg = 0.;
            //for (int i=0; i<times; i++){
            //    //int32_t* node_map_ptr = static_cast<int32_t*>(node_map->data);
            //    //// Reset the number of nodes
            //    //num_nodes = node_map.GetSize()/sizeof(int32_t);
            //    auto ret =  launch_fa_large_feature<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 0, nullptr);
            //    if (i >= warm_up_times) {
            //        avg += ret;
            //    }
            //}
            //LOG(INFO) << "On average FA takes: " << avg/(times-warm_up_times)  << " us";
            avg = 0.;
            for (int i=0; i<times; i++){
                //int32_t* node_map_ptr = static_cast<int32_t*>(node_map->data);
                //// Reset the number of nodes
                //num_nodes = node_map.GetSize()/sizeof(int32_t);
                auto ret = launch_fa_large_feature<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 2, nullptr);
                if (i >= warm_up_times) {
                    avg += ret;
                }
            }
            LOG(INFO) << "On average FA + atomic takes: " << avg/(times-warm_up_times)  << " us";
            avg = 0.;
            for (int i=0; i<times; i++){
                //int32_t* node_map_ptr = static_cast<int32_t*>(node_map->data);
                //// Reset the number of nodes
                //num_nodes = node_map.GetSize()/sizeof(int32_t);
                auto ret = launch_fa_large_feature<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 3, nullptr);
                if (i >= warm_up_times) {
                    avg += ret;
                }
            }
            LOG(INFO) << "On average FA + dynamic takes: " << avg/(times-warm_up_times)  << " us";
        }
        //avg = 0.;
        //for (int i=0; i<times; i++){
        //    num_nodes = deg_inc_node_map.GetSize()/sizeof(int32_t);
        //    int32_t* node_map_ptr = static_cast<int32_t*>(deg_inc_node_map->data);
        //    auto ret = launch_fa_large_feature<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 0, node_map_ptr);
        //    if (i >= warm_up_times) {
        //        avg += ret;
        //    }
        //}
        //LOG(INFO) << "On average FA static blk id deg_increase access takes: " << avg/(times-warm_up_times)  << " us";
        //avg = 0.;
        //for (int i=0; i<times; i++){
        //    num_nodes = deg_inc_node_map.GetSize()/sizeof(int32_t);
        //    int32_t* node_map_ptr = static_cast<int32_t*>(deg_inc_node_map->data);
        //    auto ret = launch_fa_large_feature<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 3, node_map_ptr);
        //    if (i >= warm_up_times) {
        //        avg += ret;
        //    }
        //}
        //LOG(INFO) << "On average FA with dynamic blk but no atomic and deg_increase access takes: " << avg/(times-warm_up_times)  << " us";
        if (feat_len < 64) {
            avg = 0.;
            for (int i=0; i<times; i++){
                //num_nodes = deg_inc_node_map.GetSize()/sizeof(int32_t);
                //int32_t* node_map_ptr = static_cast<int32_t*>(deg_inc_node_map->data);
                auto ret =  launch_fa_small<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 2, nullptr);
                if (i >= warm_up_times) {
                    avg += ret;
                }
            }
            LOG(INFO) << "On average FA + atomic takes: " << avg/(times-warm_up_times)  << " us";
            avg = 0.;
            for (int i=0; i<times; i++){
                //num_nodes = node_map.GetSize()/sizeof(int32_t);
                //int32_t* node_map_ptr = static_cast<int32_t*>(node_map->data);
                auto ret =  launch_fa_small<int32_t>(csr, feat_ptr, feat_len, num_nodes, num_edges, 3, nullptr);
                if (i >= warm_up_times) {
                    avg += ret;
                }
            }
            LOG(INFO) << "On average FA + dynamic takes: " << avg/(times-warm_up_times)  << " us";
        }
    }

template<typename Idx>
void printNDArray(const runtime::NDArray& arr, std::string arr_name = "arr") {
    auto len = arr->shape[0];
    auto arr_data = static_cast<Idx*> (arr->data);
    std::string print_str;
    for (int i=0; i<len; i++){
        print_str += std::to_string(arr_data[i]) + " ";
    }
    LOG(INFO) << arr_name << ":\n" << print_str;
}

template<typename Idx, typename DType>
__global__ void RgcnLayer0KernelImpl(Idx* ranges, Idx* src_ids, Idx* eids, Idx* types, DType* weight, DType* norm, DType* ret, Idx num_nodes, Idx feat_len, Idx ntypes) {
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        for (;tx<feat_len; tx += blockDim.x) {
            DType agg_val = 0.;
            for(;beg<end;beg++) {
                Idx src_id = __ldg(src_ids + beg);
                Idx eid = __ldg(eids + beg);
                Idx type_id = __ldg(types + beg);
                DType w = __ldg(weight + type_id*ntypes*feat_len + src_id*feat_len + tx);
                DType n = __ldg(norm + eid);
                agg_val += w * n;
                //printf("w:%f norm:%f agg_val:%f\n", w, n, agg_val);
            }
            ret[blockIdx.x*feat_len + tx] = agg_val;
        }
    }
}

void print_dims(const runtime::NDArray& arr) {
    std::string print_str;
    for(int i=0; i<arr->ndim; ++i) {
        print_str +=  std::to_string(arr->shape[i]) + " ";
    }
    LOG(INFO) << print_str; 
}

void RgcnLayer0Impl(
    GraphRef graph,
    runtime::NDArray weight,
    runtime::NDArray norm,
    runtime::NDArray ret){
        //LOG(INFO) << "Calling implementation of rgn layer 0 forward";
        typedef int32_t Idx;
        typedef float DType;
        auto csr = graph->GetCsrSortedByEdgeType(false);
        auto ranges = csr[0];
        auto ids = csr[1];
        auto eids = csr[2];
        auto type_ids = csr[3];
        auto range_data = static_cast<Idx*> (ranges->data);
        auto ids_data = static_cast<Idx*> (ids->data);
        auto eids_data = static_cast<Idx*> (eids->data);
        auto typeids_data = static_cast<Idx*> (type_ids->data);
        auto weight_data = static_cast<DType*> (weight->data);
        auto norm_data = static_cast<DType*> (norm->data);
        auto ret_data = static_cast<DType*> (ret->data);
        //print_dims(weight);
        //print_dims(norm);
        //print_dims(ret);
        Idx num_nodes = ranges->shape[0] - 1;
        Idx num_edges = eids->shape[0];
        Idx ntypes = weight->shape[1];
        Idx feat_len = weight->shape[2];
        //LOG(INFO) << "num edges:" << num_edges << " num nodes:" <<num_nodes;  
        int nblks = num_nodes;
        int nthrs = feat_len;
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        RgcnLayer0KernelImpl<Idx, DType><<<nblks, nthrs, 0, thr_entry->stream>>>
            (range_data, ids_data, eids_data, typeids_data, weight_data, norm_data, ret_data, num_nodes, feat_len, ntypes);
        //printNDArray<Idx>(ranges, "range");
        //printNDArray<Idx>(ids, "ids");
        //printNDArray<Idx>(eids, "eids");
        //printNDArray<Idx>(type_ids, "type_ids");
    }

template<typename Idx, typename DType>
__global__ void RgcnLayer0BackwardKernelImpl(Idx* ranges, 
  Idx* dst_ids, 
  Idx* eids, 
  Idx* types, 
  DType* grad_out, 
  DType* norm, 
  DType* grad_weight, 
  Idx num_nodes, 
  Idx feat_len, 
  Idx ntypes) {
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        for (;tx<feat_len; tx += blockDim.x) {
            for(;beg<end;beg++) {
                Idx dst_id = __ldg(dst_ids + beg);
                Idx eid = __ldg(eids + beg);
                Idx type_id = __ldg(types + beg);
                DType w = __ldg(grad_out + dst_id*feat_len + tx);
                DType n = __ldg(norm + eid);
                grad_weight[type_id * ntypes * feat_len + blockIdx.x * feat_len + tx] = w * n;
            }
        }
    }
}

void RgcnLayer0BackwardImpl(
    GraphRef graph,
    runtime::NDArray grad_out,
    runtime::NDArray norm,
    runtime::NDArray ret){
        //LOG(INFO) << "Calling implementation of rgn layer 0 backward";
        //cudaDeviceSynchronize();
        //auto t1 = std::chrono::steady_clock::now();
        typedef int32_t Idx;
        typedef float DType;
        auto csr = graph->GetCsrSortedByEdgeType(true);
        auto ranges = csr[0];
        auto ids = csr[1];
        auto eids = csr[2];
        auto type_ids = csr[3];
        auto range_data = static_cast<Idx*> (ranges->data);
        auto ids_data = static_cast<Idx*> (ids->data);
        auto eids_data = static_cast<Idx*> (eids->data);
        auto typeids_data = static_cast<Idx*> (type_ids->data);
        auto grad_out_data = static_cast<DType*> (grad_out->data);
        auto norm_data = static_cast<DType*> (norm->data);
        auto ret_data = static_cast<DType*> (ret->data);
        //print_dims(grad_out);
        //print_dims(norm);
        //print_dims(ret);
        Idx num_nodes = ranges->shape[0] - 1;
        Idx num_edges = eids->shape[0];
        Idx ntypes = ret->shape[1];
        Idx feat_len = ret->shape[2];
        int nblks = num_nodes;
        int nthrs = feat_len;
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        RgcnLayer0BackwardKernelImpl<Idx, DType><<<nblks, nthrs, 0, thr_entry->stream>>>
            (range_data, ids_data, eids_data, typeids_data, grad_out_data, norm_data, ret_data, num_nodes, feat_len, ntypes);
        //cudaDeviceSynchronize();
        //auto t2 = std::chrono::steady_clock::now();
        //LOG(INFO) << "layer 0 backward kernel takes:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -t1).count()/1000.0 << " s";
    }

template<typename Idx, typename DType>
// bgs: 0.019
__global__ void RgcnLayer1KernelImplAtomic(Idx* ranges, 
  Idx* src_ids, 
  Idx* eids, 
  Idx* types, 
  DType* hidden, 
  DType* weight, 
  DType* norm, 
  DType* ret, 
  Idx num_nodes, 
  Idx feat_len_y, 
  Idx feat_len_x, 
  Idx ntypes) {
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        Idx ty = threadIdx.x / feat_len_x;
        Idx th = threadIdx.x % feat_len_x;
        for(;beg<end;beg++) {
            Idx src_id = __ldg(src_ids + beg);
            Idx eid = __ldg(eids + beg);
            Idx type_id = __ldg(types + beg);
            DType h = __ldg(hidden + src_id*feat_len_y + ty);
            DType w = __ldg(weight + type_id*feat_len_y*feat_len_x + tx);
            DType n = __ldg(norm + eid);
            atomicAdd(ret + blockIdx.x*feat_len_x + th, w*h*n);
        }
    }
}

// bgs:
template<typename Idx, typename DType>
__global__ void RgcnLayer1KernelImpl(const Idx* ranges, 
  const Idx* src_ids, 
  const Idx* eids, 
  const Idx* types, 
  const DType* hidden, 
  const DType* weight, 
  const DType* norm, 
  DType* ret, 
  Idx num_nodes, 
  Idx feat_len_y, 
  Idx feat_len_x, 
  Idx ntypes) {
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        Idx ty = threadIdx.x / feat_len_x;
        Idx th = threadIdx.x % feat_len_x;
        DType agg_val = 0.; 
        DType w = 0.;
        Idx cur_type_id = -1;
        for(;beg<end;beg++) {
            Idx src_id = __ldg(src_ids + beg);
            Idx eid = __ldg(eids + beg);
            Idx type_id = __ldg(types + beg);
            if (type_id != cur_type_id) {
                w = __ldg(weight + type_id*feat_len_y*feat_len_x + tx);
            }
            DType h = __ldg(hidden + src_id*feat_len_y + ty);
            DType n = __ldg(norm + eid);
            agg_val += h * w * n;
        }
        atomicAdd(ret + blockIdx.x*feat_len_x + th, agg_val);
    }
}

void RgcnLayer1Impl(
    GraphRef graph,
    runtime::NDArray hidden,
    runtime::NDArray weight,
    runtime::NDArray norm,
    runtime::NDArray ret){
        //LOG(INFO) << "Calling implementation of rgn layer 1 forward";
        typedef int32_t Idx;
        typedef float DType;
        auto csr = graph->GetCsrSortedByEdgeType(false);
        auto ranges = csr[0];
        auto ids = csr[1];
        auto eids = csr[2];
        auto type_ids = csr[3];
        auto range_data = static_cast<Idx*> (ranges->data);
        auto ids_data = static_cast<Idx*> (ids->data);
        auto eids_data = static_cast<Idx*> (eids->data);
        auto typeids_data = static_cast<Idx*> (type_ids->data);
        auto hidden_data = static_cast<DType*> (hidden->data);
        auto weight_data = static_cast<DType*> (weight->data);
        auto norm_data = static_cast<DType*> (norm->data);
        auto ret_data = static_cast<DType*> (ret->data);
        //print_dims(hidden);
        //print_dims(weight);
        //print_dims(norm);
        //print_dims(ret);
        Idx num_nodes = ranges->shape[0] - 1;
        Idx num_edges = eids->shape[0];
        Idx ntypes = weight->shape[0];
        Idx feat_len_y = weight->shape[1];
        Idx feat_len_x = weight->shape[2];
        int nblks = num_nodes;
        int nthrs = feat_len_y * feat_len_x;
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        RgcnLayer1KernelImpl<Idx, DType><<<nblks, nthrs, 0, thr_entry->stream>>>
            (range_data, ids_data, eids_data, typeids_data, hidden_data, weight_data, norm_data, ret_data, num_nodes, feat_len_y, feat_len_x, ntypes);
    }

template<typename Idx, typename DType>
__global__ void RgcnLayer1BackwardKernelImpl(Idx* ranges, 
  Idx* dst_ids, 
  Idx* eids, 
  Idx* types, 
  DType* hidden, 
  DType* weight, 
  DType* norm, 
  DType* grad_out, 
  DType* grad_hidden, 
  DType* grad_weight, 
  Idx num_nodes, 
  Idx feat_len_y, 
  Idx feat_len_x, 
  Idx ntypes) {
    if (blockIdx.x < num_nodes) {
        Idx beg = __ldg(ranges + blockIdx.x);
        Idx end = __ldg(ranges + blockIdx.x + 1);
        Idx tx = threadIdx.x;
        for (;tx<feat_len_x * feat_len_y; tx += blockDim.x) {
            Idx ty = tx / feat_len_x;
            Idx th = tx % feat_len_x;
            DType h = __ldg(hidden + blockIdx.x*feat_len_y + ty);
            DType agg = 0.;
            for(;beg<end;beg++) {
                Idx dst_id = __ldg(dst_ids + beg);
                Idx eid = __ldg(eids + beg);
                Idx type_id = __ldg(types + beg);
                DType g = __ldg(grad_out + dst_id * feat_len_x + th);
                DType w = __ldg(weight + type_id*feat_len_y*feat_len_x + tx);
                DType n = __ldg(norm + eid);
                agg += g*w*n;
                atomicAdd(grad_weight + type_id*feat_len_y*feat_len_x + tx, g*h*n);
            }
            atomicAdd(grad_hidden + blockIdx.x*feat_len_y + ty, agg);
        }
    }
}

void RgcnLayer1BackwardImpl(
    GraphRef graph,
    runtime::NDArray hidden,
    runtime::NDArray weight,
    runtime::NDArray norm,
    runtime::NDArray grad_out,
    runtime::NDArray grad_hidden,
    runtime::NDArray grad_weight){
        //cudaDeviceSynchronize();
        //auto t1 = std::chrono::steady_clock::now();
        typedef int32_t Idx;
        typedef float DType;
        auto csr = graph->GetCsrSortedByEdgeType(true);
        auto ranges = csr[0];
        auto ids = csr[1];
        auto eids = csr[2];
        auto type_ids = csr[3];
        auto range_data = static_cast<Idx*> (ranges->data);
        auto ids_data = static_cast<Idx*> (ids->data);
        auto eids_data = static_cast<Idx*> (eids->data);
        auto typeids_data = static_cast<Idx*> (type_ids->data);
        auto hidden_data = static_cast<DType*> (hidden->data);
        auto weight_data = static_cast<DType*> (weight->data);
        auto norm_data = static_cast<DType*> (norm->data);
        auto grad_out_data = static_cast<DType*> (grad_out->data);
        auto grad_hidden_data = static_cast<DType*> (grad_hidden->data);
        auto grad_weight_data = static_cast<DType*> (grad_weight->data);
        //print_dims(hidden);
        //print_dims(weight);
        //print_dims(norm);
        //print_dims(grad_out);
        //print_dims(grad_hidden);
        //print_dims(grad_weight);
        Idx num_nodes = ranges->shape[0] - 1;
        Idx num_edges = eids->shape[0];
        Idx ntypes = weight->shape[0];
        Idx feat_len_y = weight->shape[1];
        Idx feat_len_x = weight->shape[2];
        int nblks = num_nodes;
        int nthrs = feat_len_y * feat_len_x;
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        
        RgcnLayer1BackwardKernelImpl<Idx, DType>
            <<<nblks, nthrs, 0, thr_entry->stream>>>
            (range_data, ids_data, eids_data, typeids_data,
             hidden_data, weight_data, norm_data, grad_out_data, grad_hidden_data, grad_weight_data,
             num_nodes, feat_len_y, feat_len_x, ntypes);
        //cudaDeviceSynchronize();
        //auto t2 = std::chrono::steady_clock::now();
        //LOG(INFO) << "layer 1 backward kernel takes:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -t1).count()/1000.0 << " s";
    }

void BackwardFusedGatKernelImpl(
    const CSRWrapper& graph,
    runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    runtime::NDArray grad_out,
    runtime::NDArray grad_feat_src,
    runtime::NDArray grad_el,
    runtime::NDArray grad_er,
    float slope) {
        typedef int32_t Idx;
        typedef float DType;
        const Idx MAX_NBLKS = 65535;
        const Idx MAX_NTHRS = 1024;
        // zero out ret, and packing feat_src, el, er, ret, graph together into one struct using raw float pointers
        // get csr matrix
        BackwardGatFusedData<Idx, DType> gdata;
        int64_t el_xlen =  utils::ComputeXLength(el);
        int64_t feat_src_xlen =  utils::ComputeXLength(feat_src);
        gdata.feat_src = static_cast<DType*>(feat_src->data);
        gdata.el = static_cast<DType*>(el->data);
        gdata.er = static_cast<DType*>(er->data);
        gdata.sum = static_cast<DType*>(sum->data);
        gdata.exp = static_cast<DType*>(exp->data);
        gdata.ret = static_cast<DType*>(ret->data);
        gdata.grad_out= static_cast<DType*>(grad_out->data);
        gdata.grad_feat_src = static_cast<DType*>(grad_feat_src->data);
        gdata.grad_el = static_cast<DType*>(grad_el->data);
        gdata.grad_er = static_cast<DType*>(grad_er->data);
        gdata.leaky_relu_slope = slope;
        gdata.n = el.GetSize()/sizeof(DType)/el_xlen; 
        gdata.e_xlen = el_xlen;
        gdata.feat_src_xlen =  feat_src_xlen;
        gdata.feat_src_hidden = feat_src_xlen/el_xlen;
        auto outcsr = graph.GetOutCSRMatrix();
        minigun::Csr<Idx> ocsr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
        gdata.eids = static_cast<Idx*>(outcsr.data->data);
        // write a device function and call it from here
        //LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" << feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen 
        //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  << " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
        //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph csr column indices length:" << csr.column_indices.length;
        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,grad_out,ocsr,el_xlen, feat_src_xlen, gdata.eids);
        // Configure kernel launch parameters.
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        int nthrs_x = utils::FindNumThreads(el_xlen, 64);
        int nthrs_y = utils::FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS/nthrs_x);
        int nblks_x = 1;
        int nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nthrs(nthrs_x, nthrs_y);
        const dim3 nblks(nblks_x, nblks_y);
        //LOG(INFO) << "GradFeatSrc kernel blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        fusedGatBackwardGradFeatSrc<<<nblks, nthrs, 0, thr_entry->stream>>>(gdata, ocsr);
        //const dim3 nthrs3(nthrs_y, nthrs_x);
        //fusedGatBackwardGradElEr4<<<nblks, nthrs3, 0, thr_entry->stream>>>(gdata, ocsr);
        fusedGatBackwardGradElEr<<<nblks, nthrs, 0, thr_entry->stream>>>(gdata, ocsr);
}


template void BinaryReduceImpl<kDLGPU>(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

template void BinaryReduceBcastImpl<kDLGPU>(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

template void BackwardBinaryReduceImpl<kDLGPU>(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping, NDArray rhs_mapping, NDArray out_mapping,
    NDArray lhs_data, NDArray rhs_data, NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_lhs_data, NDArray grad_rhs_data);

template void BackwardBinaryReduceBcastImpl<kDLGPU>(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs_tgt, binary_op::Target rhs_tgt,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs, runtime::NDArray rhs, runtime::NDArray out, runtime::NDArray grad_out,
    runtime::NDArray grad_lhs, runtime::NDArray grad_rhs);

template <typename Idx>
std::string print_csr(const minigun::Csr<Idx>& csr) {
    Idx row_len = csr.row_offsets.length;
    Idx col_ind_len = csr.column_indices.length;
    Idx* row_cpu = (Idx*) malloc(sizeof(Idx)*csr.row_offsets.length);
    Idx* col_ind_cpu = (Idx*) malloc(sizeof(Idx)*csr.column_indices.length);
    cudaMemcpy(row_cpu, csr.row_offsets.data, sizeof(Idx)*csr.row_offsets.length, cudaMemcpyDeviceToHost);
    cudaMemcpy(col_ind_cpu, csr.column_indices.data, sizeof(Idx)*csr.column_indices.length, cudaMemcpyDeviceToHost);
    std::string tmp = "";
    tmp += "row_offsets:\n";
    for (Idx i=0; i<row_len;++i) {
        tmp += std::to_string(row_cpu[i]) + ", ";
    }
    tmp += "\ncol_indices:\n";
    for (Idx i=0; i<col_ind_len; ++i) {
        tmp += std::to_string(col_ind_cpu[i])  + ", ";
    }
    tmp+= "\n";
    free(row_cpu);
    free(col_ind_cpu);
    return tmp;
}

template <typename Idx, typename DType>
std::string print_gdata2d(runtime::NDArray a, Idx dim1, Idx dim2) {
    if (a->ctx.device_type != kDLGPU) {
        LOG(FATAL) << "Tensor is not on GPU it is on:" << a->ctx.device_type;
    }
    Idx size = a.GetSize();
    DType* vals = (DType*)malloc(size);
    memset(vals, 0, size);
    cudaError_t err = cudaMemcpy(vals, a->data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG(FATAL) << std::string(cudaGetErrorString(err));
    }
    Idx n = size/sizeof(DType);
    if (n != dim1 * dim2) {
        LOG(FATAL) << "dim1 * dim2 != n, dim1:" << dim1 << " dim2:" << dim2 << " n" << n;
    }
    std::string str = "[";
    for (Idx i=0; i<dim1; i++) {
        str += "[";
        for (Idx j=0; j<dim2; j++) {
            str += std::to_string(vals[i*dim2+j]) + ", ";
        }
        str += "]\n";
    }
    str += "]\n";
    free(vals);
    return str;
}

template <typename Idx, typename DType>
void print_gdata(runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    const minigun::Csr<Idx> &csr,
    Idx el_xlen,
    Idx feat_src_xlen,
    runtime::NDArray  eid) {
        std::string str_csr = print_csr<Idx>(csr);
        std::string str_el = print_gdata2d<Idx, DType>(el, csr.row_offsets.length-1, el_xlen);
        std::string str_er = print_gdata2d<Idx, DType>(er, csr.row_offsets.length-1, el_xlen);
        std::string str_feat_src= print_gdata2d<Idx, DType>(feat_src, csr.row_offsets.length-1, feat_src_xlen);
        std::string str_exp = print_gdata2d<Idx, DType>(exp, csr.column_indices.length, el_xlen);
        std::string str_sum = print_gdata2d<Idx, DType>(sum, csr.row_offsets.length-1, el_xlen);
        std::string str_ret = print_gdata2d<Idx, DType>(ret, csr.row_offsets.length-1, feat_src_xlen);
        std::string str_eid = print_gdata2d<Idx, Idx>(eid, csr.column_indices.length, 1);
        LOG(INFO) << "csr " << str_csr << " feat_src "<< str_feat_src << " el "<< str_el << " er " << str_er << "exp " << str_exp << "sum " <<str_sum << " ret" << str_ret << " eid"<<str_eid;
}

}  // namespace kernel
}  // namespace dgl
