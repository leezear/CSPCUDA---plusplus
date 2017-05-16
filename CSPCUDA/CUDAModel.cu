#pragma once
#include "CUDAModel.cuh"
#include <thrust\reduce.h>
#include <thrust\scan.h>
#include <thrust\transform.h>
#include <thrust\pair.h>
#include <thrust\reduce.h>
#include <thrust\sort.h>
#include <thrust\unique.h>
#include <thrust\functional.h>
#include <thrust\for_each.h>
#include <thrust\system\cuda\execution_policy.h>

extern "C" bool DataTransfer(XMLModel *model);
extern "C" bool BuildArcsModel(XMLModel *model);
extern "C" int AC4GpuPlusInitialization();
extern "C" int AC4GpuPropagation();

#define CSCOUNT 3
typedef thrust::tuple<int, int> Node;
typedef thrust::device_vector<Node> Nodes;
typedef thrust::device_vector<int>::iterator   IntIterator;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator> D_ArcIterTuple;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator, IntIterator, IntIterator> D_ArcSorcIterTuple;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator> D_CounterIterTuple;
typedef thrust::tuple<IntIterator, IntIterator> D_NodesIterTuple;
typedef thrust::zip_iterator<D_ArcIterTuple> D_ArcTupleIter;
typedef thrust::zip_iterator<D_CounterIterTuple> D_CounterIter;
typedef thrust::zip_iterator<D_NodesIterTuple> D_NodesIter;
typedef thrust::zip_iterator<D_ArcSorcIterTuple> D_ArcSorcIter;
typedef thrust::pair<D_CounterIter, IntIterator> Counter;
const static int MAXTHREADSPERBLOCK = 1024;
const static int MaxThreads = 10240000;
int dbsum;
int stream_size;
bool propagationEnable = true;
thrust::device_vector<int> d_vars_size;
thrust::host_vector<int> h_vars_size;
//点偏移量
thrust::device_vector<int> d_node_global;
thrust::device_vector<int> d_nodes_set;
//thrust::device_vector<int> d_segment_indexes;

//counter偏移量
//thrust::device_vector<int> d_local_counter;
//counter总数
int counter_sum;
int nodes_sum;
__device__ bool d_is_sat;
__device__ bool d_conti;

struct D_Arcs
{
	thrust::device_vector<int> d_vars0;
	thrust::device_vector<int> d_vals0;
	thrust::device_vector<int> d_vars1;
	thrust::device_vector<int> d_vals1;
	thrust::device_vector<int> d_sorcs;
	thrust::device_vector<int> d_cmaps;

	D_Arcs()
	{
	}

	D_Arcs(size_t len)
	{
		resize(len);
	}

	void operator()(size_t len)
	{
		resize(len);
	}

	void resize(size_t len)
	{
		d_vars0.resize(len);
		d_vals0.resize(len);
		d_vars1.resize(len);
		d_vals1.resize(len);
		d_sorcs.resize(len);
		d_cmaps.resize(len);
	}

	//void erase()

}d_arcs, d_arcs2;

struct D_Vars_Size
{
	thrust::device_vector<int> vars;
	thrust::device_vector<int> sizes;
	D_Vars_Size(){}
	D_Vars_Size(size_t len)
	{
		resize(len);
	}

	void resize(size_t len)
	{
		vars.resize(len);
		sizes.resize(len);
	}
};

struct H_Arcs
{
	thrust::host_vector<int> h_vars0;
	thrust::host_vector<int> h_vals0;
	thrust::host_vector<int> h_vars1;
	thrust::host_vector<int> h_vals1;
	thrust::host_vector<int> h_sorcs;
	thrust::host_vector<int> h_cmaps;

	void operator= (D_Arcs das)
	{
		h_vars0 = das.d_vars0;
		h_vals0 = das.d_vals0;
		h_vars1 = das.d_vars1;
		h_vals1 = das.d_vals1;
		h_sorcs = das.d_sorcs;
		h_cmaps = das.d_cmaps;
	}
}h_arcs;

struct D_Node
{
	thrust::device_vector<int> vars;
	thrust::device_vector<int> vals;
	D_Node()
	{
	}

	D_Node(size_t len)
	{
		vars.resize(len);
		vals.resize(len);
	}

	void operator()(size_t len)
	{
		resize(len);
	}

	void resize(size_t len)
	{
		vars.resize(len);
		vals.resize(len);
	}
}d_nodes;

struct 	D_SegmentIndex
{
	thrust::device_vector<int> start;
	thrust::device_vector<int> end;

	D_SegmentIndex()
	{
	}

	D_SegmentIndex(size_t len)
	{
		resize(len);
	}

	D_SegmentIndex(size_t len, int init_value)
	{
		resize(len, init_value);
	}

	void resize(size_t len)
	{
		start.resize(len);
		end.resize(len);
	}

	void resize(size_t len, int init_value)
	{
		start.resize(len, init_value);
		end.resize(len, init_value);
	}
}d_segidx;

__global__ void ComputeLocalOffset(int *d_offset, int *d_local_counter, XMLConstraint *d_c, XMLDomain *d_d, XMLVariable *d_v)
{
	int i = threadIdx.x;
	int x = d_c[i].scope.x;
	int y = d_c[i].scope.y;
	int x_dm_id = d_v[x].dm_id;
	int y_dm_id = d_v[y].dm_id;
	int x_dm_size = d_d[x_dm_id].size;
	int y_dm_size = d_d[y_dm_id].size;
	int local_offset = x_dm_size*y_dm_size;
	d_offset[i] = 2 * local_offset;
	d_local_counter[i] = x_dm_size + y_dm_size;
}

__global__ void GenerateVars_size(int *vars_size, XMLVariable *vars, XMLDomain *dms)
{
	int i = threadIdx.x;
	int dm_id = vars[i].dm_id;
	int dm_size = dms[dm_id].size;
	vars_size[i] = dm_size;
}

__global__ void GenerateNodes(int *node_var, int *node_val, int *offset, int var_id)
{
	int i = threadIdx.x;
	int idx = offset[var_id] + i;
	node_var[idx] = var_id;
	node_val[idx] = i;
}

__global__ void GenerateNodesLaunch(int *node_var, int *node_val, int *offset, XMLVariable *vars, XMLDomain *dms)
{
	int i = threadIdx.x;
	int dm_id = vars[i].dm_id;
	int dm_size = dms[dm_id].size;
	GenerateNodes << <1, dm_size >> >(node_var, node_val, offset, i);
}

__global__
void BuildArc(
int *d_vars0,
int *d_vals0,
int *d_vars1,
int *d_vals1,
int *d_sorc,
int *d_cmap,
int d_offset,
int d_global_offset,
int d_global_counter,
XMLDomain dm_0,
XMLDomain dm_1,
XMLVariable var_0,
XMLVariable var_1,
XMLConstraint cst, int sorc
)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	int val0 = dm_0.values[x];
	int val1 = dm_1.values[y];
	int var0 = var_0.id;
	int var1 = var_1.id;
	int local_arc0 = blockDim.x*x + y;
	int local_arc1 = gridDim.x*y + x;
	int global_arc0 = local_arc0 + d_global_offset;
	int global_arc1 = local_arc1 + d_offset / 2 + d_global_offset;
	int global_counter0 = d_global_counter + x;
	int global_counter1 = d_global_counter + gridDim.x + y;
	//printf("%d %d %d %d,%d %d %d\n", var0, val0, var1, val1, sorc, global_counter0, global_counter1);

	d_vars0[global_arc0] = var0;
	d_vals0[global_arc0] = val0;
	d_vars1[global_arc0] = var1;
	d_vals1[global_arc0] = val1;
	d_vars0[global_arc1] = var1;
	d_vals0[global_arc1] = val1;
	d_vars1[global_arc1] = var0;
	d_vals1[global_arc1] = val0;
	d_sorc[global_arc0] = sorc;
	d_sorc[global_arc1] = sorc;
	d_cmap[global_arc0] = global_counter0;
	d_cmap[global_arc1] = global_counter1;

	return;
}

__global__
void ModifyTuple(
int* d_sorc,
int d_offset,
int d_global_offset,
XMLDomain dm_0,
XMLDomain dm_1,
XMLRelation rel,
int semantices,
int size
)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < size)
	{
		bituple bt = rel.tuples[i];
		int var0_size = dm_0.size;
		int var1_size = dm_1.size;
		int sorc = semantices;
		int local_sorc0 = bt.x*var1_size + bt.y;
		int local_sorc1 = bt.y*var0_size + bt.x;
		int global_sorc0 = local_sorc0 + d_global_offset;
		int global_sorc1 = local_sorc1 + d_offset / 2 + d_global_offset;
		d_sorc[global_sorc0] = sorc;
		d_sorc[global_sorc1] = sorc;
	}
}

__global__
void BuildArcsLaunch(
int *d_vars0,
int *d_vals0,
int *d_vars1,
int *d_vals1,
int *d_sorc,
int *d_cmap,
int *d_offset,
int *d_global_offset,
int *d_global_counter,
XMLDomain *dms,
XMLVariable *vars,
XMLRelation *rels,
XMLConstraint *csts
)
{
	int mtpb = 1024;
	int i = threadIdx.x;
	XMLConstraint constraint = csts[i];
	int x = constraint.scope.x;
	int y = constraint.scope.y;
	XMLRelation relation = rels[constraint.re_id];
	int r_size = relation.size;
	XMLVariable var0 = vars[x];
	XMLVariable var1 = vars[y];
	int x_dm_id = var0.dm_id;
	int y_dm_id = var1.dm_id;
	XMLDomain dm_0 = dms[x_dm_id];
	XMLDomain dm_1 = dms[y_dm_id];
	int x_size = dm_0.size;
	int y_size = dm_1.size;
	int semantices = relation.semantices;
	int sorc = !semantices;
	int block_size = r_size / (mtpb)+!(!(r_size % (mtpb)));
	BuildArc << <x_size, y_size >> >(d_vars0, d_vals0, d_vars1, d_vals1, d_sorc, d_cmap, d_offset[i], d_global_offset[i], d_global_counter[i], dm_0, dm_1, var0, var1, constraint, sorc);
	__syncthreads();
	ModifyTuple << <block_size, mtpb >> >(d_sorc, d_offset[i], d_global_offset[i], dm_0, dm_1, relation, semantices, r_size);
}

__global__ void Vars_Resize(int *var, int *del, int*vars)
{
	int i = threadIdx.x;
	int del_var_id = var[i];
	int del_now = del[i];
	int size = vars[del_var_id];
	int new_size = size - del_now;
	vars[del_var_id] = new_size;
}

__global__ void show(int *a)
{
	int i = threadIdx.x;
	printf("%d\n", a[i]);
}

__global__ void CompareVar_Size(int *old_var_size, int *tmp_var_size, int size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	d_is_sat = true;
	d_conti = false;

	if (idx < size)
	{
		int old = old_var_size[idx];
		int tmp = tmp_var_size[idx];
		if (old != tmp)
		{
			d_conti = true;
		}
		if (tmp == 0)
		{
			d_is_sat = false;
		}
	}
}

struct TernaryPredicate
{
	template<typename Tuple>
	__host__ __device__ bool operator()(const Tuple& a, const Tuple& b)
	{
		return(
			(thrust::get<0>(a) == thrust::get<0>(b)) &&
			(thrust::get<1>(a) == thrust::get<1>(b)) &&
			(thrust::get<2>(a) == (thrust::get<2>(b)))
			);
	}
};

struct is_conflict
{
	template<typename Tuple>
	//__host__ __device__ bool operator()(const thrust::tuple<const int&, const int&, const int&, const int&, const int&, const int&> &a)
	__host__ __device__ bool operator()(const Tuple& a)
	{
		return (!(thrust::get<4>(a)));
	}
};

struct is_zero
{
	int *nodes;
	int *offset;
	int len;
	is_zero(int *nodes, int *offset, int len) :len(len), nodes(nodes), offset(offset){}
	is_zero(){}
	__host__ __device__
		bool operator()(const int s_val) const {
			return (!s_val);
		}
};

struct ModifyNodes
{
	int *nodes;
	int *node_offset;
	ModifyNodes(int *nodes, int *node_offset) :nodes(nodes), node_offset(node_offset){}

	template<typename Tuple>
	__host__ __device__	void operator()(const Tuple &t)
	{
		int var = thrust::get<0>(t);
		int val = thrust::get<1>(t);
		int sorc = thrust::get<2>(t);
		if (sorc == 0)
		{
			//thrust::get<2>(t) = 0;
			//printf("-1\n");
			int node_idx = node_offset[var] + val;
			nodes[node_idx] = 0;
		}
	}
};

//struct ModifyNodesProp
//{
//	int *nodes;
//	int *node_offset;
//	int *d_vars_size_tmp;
//	ModifyNodesProp(int *nodes, int *node_offset, int *d_vars_size_tmp) :nodes(nodes), node_offset(node_offset), d_vars_size_tmp(d_vars_size_tmp){}
//
//	template<typename Tuple>
//	__device__	void operator()(const Tuple &t)
//	{
//		int var = thrust::get<0>(t);
//		int val = thrust::get<1>(t);
//		int sorc = thrust::get<2>(t);
//		if (sorc == -1)
//		{
//			thrust::get<2>(t) = 0;
//			int node_idx = node_offset[var] + val;
//			int e = nodes[node_idx];
//			if (e == 1)
//			{
//				nodes[node_idx] = 0;
//				atomicAdd(&d_vars_size_tmp[var], -1);
//				if (d_vars_size_tmp[var] < 0)
//				{
//					d_vars_size_tmp[var] = 0;
//				}
//			}
//		}
//	}
//};

struct ModifyNodesProp
{
	int *nodes;
	int *node_offset;
	int *d_vars_size_tmp;
	ModifyNodesProp(int *nodes, int *node_offset, int *d_vars_size_tmp) :nodes(nodes), node_offset(node_offset), d_vars_size_tmp(d_vars_size_tmp){}

	template<typename Tuple>
	__device__	void operator()(const Tuple &t)
	{
		int var = thrust::get<0>(t);
		int val = thrust::get<1>(t);
		int sorc = thrust::get<2>(t);
		if (sorc == -1)
		{
			thrust::get<2>(t) = 0;
			int node_idx = node_offset[var] + val;
			int e = nodes[node_idx];
			atomicAdd(&d_vars_size_tmp[var], 0 - nodes[node_idx]);
			nodes[node_idx] = 0;
			if (d_vars_size_tmp[var] < 0)
			{
				d_vars_size_tmp[var] = 0;
			}
		}
	}
};

struct ModifyArcs
{
	int *nodes;
	int *node_offset;
	int *counter_value;
	ModifyArcs(int *nodes, int *node_offset, int *counter_value) :nodes(nodes), node_offset(node_offset), counter_value(counter_value){}

	template<typename Tuple>
	__device__	void operator()(const Tuple &t)
	{
		int var0 = thrust::get<0>(t);
		int val0 = thrust::get<1>(t);
		int var1 = thrust::get<2>(t);
		int val1 = thrust::get<3>(t);
		int sorc = thrust::get<4>(t);
		int cmap = thrust::get<5>(t);

		int s0 = nodes[node_offset[var0] + val0];
		int s1 = nodes[node_offset[var1] + val1];

		if (sorc == 1 && s0 == 0)
		{
			thrust::get<4>(t) = 0;
			return;
		}

		//bool del_node = !(nodes[node_offset[var0] + val0] && nodes[node_offset[var1] + val1]);

		if (sorc == 1 && s1 == 0)
		{
			thrust::get<4>(t) = 0;
			atomicAdd(&counter_value[cmap], -1);

			if (counter_value[cmap] == 0)
			{
				counter_value[cmap] = -1;
				//printf("counter = %d\n", counter_value[cmap]);
				//printf("00\n");
			}
			return;
		}
	}
};

struct DeleteNodes
{
	int *nodes;
	int *offset;
	int len;

	DeleteNodes(int *nodes, int *offset, int len) :len(len), nodes(nodes), offset(offset){}

	template <typename Tuple>
	__host__ __device__ void operator()(const Tuple &t)
	{
		int var = thrust::get<0>(t);
		int val = thrust::get<1>(t);
		nodes[offset[var] + val] = 0;
	}
};

struct Is_Deleted
{
	int *nodes;
	int *offset;
	int len;
	Is_Deleted(int *nodes, int *offset, int len) : nodes(nodes), offset(offset), len(len) {}

	template <typename Tuple>
	__host__ __device__ bool operator()(const Tuple &t)
	{
		int var = thrust::get<0>(t);
		int val = thrust::get<1>(t);
		int global_offset = offset[var] + val;
		bool deleted = nodes[global_offset];
		return !deleted;
	}
};

struct Build_Segment_index
{
	int *seg;
	int *g_offset;
	int *start;
	int *end;
	int arcdim;
	Build_Segment_index(int *seg, int *g_offset, int arcdim) :seg(seg), g_offset(g_offset), arcdim(arcdim){}
	Build_Segment_index(int *start, int *end, int *g_offset, int arcdim) :start(start), end(end), g_offset(g_offset), arcdim(arcdim){}
	__host__ __device__ void operator()(const int &idx)
	{
		int offset = g_offset[idx];
		int offset_pre = g_offset[idx - 1];
		int segment = offset / arcdim;
		int segment_pre = offset_pre / arcdim;

		if (segment != segment_pre)
		{
			start[segment] = offset_pre;

			if ((segment - 1) >= 0)
			{
				end[segment - 1] = offset_pre - 1;
			}
		}
	}
};

XMLDomain *h_dms;
XMLDomain *d_dms;
XMLVariable *d_vs;
XMLRelation *h_rs;
XMLRelation *d_rs;
XMLConstraint *d_cs;

extern "C" bool DataTransfer(XMLModel *model)
{
#pragma region 拷贝参数域
	int ds_size = model->ds_size;
	int ds_len = ds_size *sizeof(XMLDomain);
	int d_size;
	h_dms = new XMLDomain[ds_size];
	memcpy(h_dms, model->domains, ds_len);

	for (size_t i = 0; i < ds_size; ++i)
	{
		d_size = model->domains[i].size;
		cudaMalloc(&(h_dms[i].values), d_size*sizeof(int));
		cudaMemcpy(h_dms[i].values, model->domains[i].values, d_size*sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMalloc((void**)&d_dms, ds_len);
	cudaMemcpy(d_dms, h_dms, ds_len, cudaMemcpyHostToDevice);
	//ShowDomains << <ds_size, d_size >> >(d_dms);
#pragma endregion

#pragma region 拷贝参数数组
	int vs_size = model->vs_size;
	int vs_len = vs_size*sizeof(XMLVariable);
	//int v_size;
	cudaMalloc((void **)&d_vs, vs_len);
	cudaMemcpy(d_vs, model->variables, vs_len, cudaMemcpyHostToDevice);
	//ShowVariables << <1, vs_size >> >(d_vs);
#pragma endregion

#pragma region 拷贝关系数组
	int rs_size = model->rs_size;
	int rs_len = rs_size*sizeof(XMLRelation);
	int r_size;
	int r_maxsize = 0;
	int r_len;
	h_rs = new XMLRelation[rs_size];
	memcpy(h_rs, model->relations, rs_len);

	for (size_t i = 0; i < rs_size; i++)
	{
		r_size = model->relations[i].size;
		r_maxsize = (r_size>r_maxsize) ? r_size : r_maxsize;
		r_len = r_size*sizeof(bituple);
		cudaMalloc((void**)&(h_rs[i].tuples), r_len);
		cudaMemcpy(h_rs[i].tuples, model->relations[i].tuples, r_len, cudaMemcpyHostToDevice);
	}
	cudaMalloc((void **)&d_rs, rs_len);
	cudaMemcpy(d_rs, h_rs, rs_len, cudaMemcpyHostToDevice);
	//ShowRelations << <rs_size, r_maxsize >> >(d_rs);
#pragma endregion

#pragma region 拷贝约束
	int cs_size = model->cs_size;
	int cs_len = cs_size*sizeof(XMLConstraint);
	cudaMalloc((void **)&d_cs, cs_len);
	cudaMemcpy(d_cs, model->constraints, cs_len, cudaMemcpyHostToDevice);
	//ShowConstraints << <1, cs_size >> >(d_cs);
#pragma endregion

#pragma region 构建模型
	//cudaMemcpyToSymbol(d_ds, &ds_size, sizeof(int));
	//cudaMemcpyToSymbol(d_vs, &vs_size, sizeof(int));
	//	cudaMemcpyToSymbol
	//	cudaMemcpyToSymbol
	//cudaMemcpyToSymbol(&d_csize, &model->cs_size, sizeof(int));
	//ShowDeviceVariables << <1, 1 >> >(1);
	//XMLModel *h_model = new XMLModel;
	//XMLModel *d_model;
	//memcpy(h_model, model, sizeof(XMLModel));
	//cudaMalloc((void **)&d_model, sizeof(XMLModel));
	//cudaMalloc((void**)&h_model->domains, sizeof(h_dms));
	//cudaMalloc((void **)&h_model->variables, sizeof(d_vs));
	//cudaMalloc((void**)&h_model->variables, sizeof(h_rs));
	//cudaMalloc((void **)h_model->constraints, sizeof(d_cs));
	//cudaMemcpy(h_model->domains, h_dms, ds_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(h_model->variables, model->variables, vs_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(h_model->relations, h_rs, rs_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(h_model->constraints, model->constraints, cs_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_model, h_model, sizeof(XMLModel), cudaMemcpyHostToDevice);
	//ShowModel << <1, 1 >> >(d_model);
#pragma endregion
	return true;
}
extern "C" bool BuildArcsModel(XMLModel *model)
{
	int ds_size = model->ds_size;
	int rs_size = model->rs_size;
	int cs_size = model->cs_size;
	int vs_size = model->vs_size;
#pragma region 计算弧局部/全局偏移量
	thrust::device_vector<int> d_local_counter(cs_size);
	thrust::device_vector<int> d_global_counter(cs_size, 0);
	int *d_local_counter_ptr = thrust::raw_pointer_cast(d_local_counter.data());

	thrust::device_vector<int> d_offset(cs_size, 0);
	thrust::device_vector<int> d_global_offset(cs_size, 0);
	thrust::device_vector<int> d_offset_index(cs_size);
	thrust::sequence(d_offset_index.begin(), d_offset_index.end());

	int* d_offset_ptr = thrust::raw_pointer_cast(d_offset.data());
	ComputeLocalOffset << <1, cs_size >> >(d_offset_ptr, d_local_counter_ptr, d_cs, d_dms, d_vs);

	int sum = thrust::reduce(d_offset.begin(), d_offset.end(), (int)0, thrust::plus<int>());
	counter_sum = thrust::reduce(d_local_counter.begin(), d_local_counter.end());
	d_arcs2.resize(counter_sum);

	//printf("counter_sum = %3d\n", counter_sum);

	dbsum = sum;
	//printf("edge = %d\n", dbsum / 2);
	int *d_global_offset_ptr = thrust::raw_pointer_cast(d_global_offset.data());
	int *d_global_counter_ptr = thrust::raw_pointer_cast(d_global_counter.data());
	thrust::exclusive_scan(d_offset.begin(), d_offset.end(), d_global_offset.begin());
	thrust::exclusive_scan(d_local_counter.begin(), d_local_counter.end(), d_global_counter.begin());

	d_arcs(dbsum);
#pragma endregion

#pragma region 创建var_size数组
	d_vars_size.resize(vs_size);
	int* d_vars_size_ptr = thrust::raw_pointer_cast(d_vars_size.data());
	//var_size *d_vars_size_ptr = thrust::raw_pointer_cast(d_vars_size.data());
	GenerateVars_size << <1, vs_size >> >(d_vars_size_ptr, d_vs, d_dms);
	//h_vars_size = d_vars_size;
#pragma endregion

#pragma region 创建arcs数组
	int *d_vars0_ptr = thrust::raw_pointer_cast(d_arcs.d_vars0.data());
	int *d_vals0_ptr = thrust::raw_pointer_cast(d_arcs.d_vals0.data());
	int *d_vars1_ptr = thrust::raw_pointer_cast(d_arcs.d_vars1.data());
	int *d_vals1_ptr = thrust::raw_pointer_cast(d_arcs.d_vals1.data());
	int *d_sorcs_ptr = thrust::raw_pointer_cast(d_arcs.d_sorcs.data());
	int *d_cmaps_ptr = thrust::raw_pointer_cast(d_arcs.d_cmaps.data());

	BuildArcsLaunch << <1, cs_size >> >(
		d_vars0_ptr,
		d_vals0_ptr,
		d_vars1_ptr,
		d_vals1_ptr,
		d_sorcs_ptr,
		d_cmaps_ptr,
		d_offset_ptr,
		d_global_offset_ptr,
		d_global_counter_ptr,
		d_dms,
		d_vs,
		d_rs,
		d_cs
		);

	/*int startindex = 0;
	int endindex = 0;
	h_arcs = d_arcs;

	std::cout << "input index range:" << std::endl;
	scanf("%d %d", &startindex, &endindex);
	while (!((startindex == -1) && (endindex == -1)))
	{
	if ((startindex == 0) && (endindex == 0))
	{
	for (size_t i = 0; i < dbsum; ++i)
	{
	printf("%4d:(%d,%d)--(%d,%d)=%d~%d\n",
	i,
	h_arcs.h_vars0[i],
	h_arcs.h_vals0[i],
	h_arcs.h_vars1[i],
	h_arcs.h_vals1[i],
	h_arcs.h_sorcs[i],
	h_arcs.h_cmaps[i]
	);
	}
	}
	for (size_t i = startindex; i < endindex; ++i)
	{
	printf("%4d:(%d,%d)--(%d,%d)=%d~%d\n",
	i,
	h_arcs.h_vars0[i],
	h_arcs.h_vals0[i],
	h_arcs.h_vars1[i],
	h_arcs.h_vals1[i],
	h_arcs.h_sorcs[i],
	h_arcs.h_cmaps[i]
	);
	}
	std::cout << "input index range:" << std::endl;
	scanf("%d %d", &startindex, &endindex);
	}*/
#pragma endregion

#pragma region 计算点局部/全局偏移量
	nodes_sum = thrust::reduce(d_vars_size.begin(), d_vars_size.end());
	//printf("node = %d\n", nodes_sum);
	d_node_global.resize(vs_size);
	thrust::exclusive_scan(d_vars_size.begin(), d_vars_size.end(), d_node_global.begin());
	d_nodes_set.resize(nodes_sum, 1);
	d_nodes.resize(nodes_sum);
	int *var_ptr = thrust::raw_pointer_cast(d_nodes.vars.data());
	int *val_ptr = thrust::raw_pointer_cast(d_nodes.vals.data());
	int *nodes_offset = thrust::raw_pointer_cast(d_node_global.data());
	GenerateNodesLaunch << <1, vs_size >> >(var_ptr, val_ptr, nodes_offset, d_vs, d_dms);
#pragma endregion

#pragma region 释放显存/内存变量
	for (size_t i = 0; i < ds_size; i++)
	{
		cudaFree(h_dms[i].values);
	}

	cudaFree(d_dms);
	delete[]h_dms;
	h_dms = NULL;

	cudaFree(d_vs);

	for (size_t i = 0; i < rs_size; i++)
	{
		cudaFree(h_rs[i].tuples);
	}
	cudaFree(d_rs);
	delete[] h_rs;
	h_rs = NULL;
#pragma endregion

	return true;
}

extern "C" int AC4GpuPlusInitialization()
{
#pragma region 初始化数据结构
	int i;
	int *d_vars_key_ptr;
	int *d_vars_size_ptr;
	int vs_size = d_vars_size.size();
	int *nodes = thrust::raw_pointer_cast(d_nodes_set.data());
	int *nodes_offset = thrust::raw_pointer_cast(d_node_global.data());
	int *counter_value = thrust::raw_pointer_cast(d_arcs2.d_sorcs.data());
	cudaStream_t cs[CSCOUNT];
	thrust::device_vector<int> d_vars_key(vs_size);
	D_ArcSorcIter new_end;
	d_vars_size_ptr = thrust::raw_pointer_cast(d_vars_size.data());
	d_vars_key_ptr = thrust::raw_pointer_cast(d_vars_key.data());

	for (i = 0; i < CSCOUNT; ++i)
	{
		cudaStreamCreate(&(cs[i]));
	}

#pragma endregion

	thrust::reduce_by_key(
		thrust::cuda::par.on(cs[0]),
		thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.begin(), d_arcs.d_vals0.begin(), d_arcs.d_vars1.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.end(), d_arcs.d_vals0.end(), d_arcs.d_vars1.end())),
		d_arcs.d_sorcs.begin(),
		thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.begin(), d_arcs2.d_vals0.begin(), d_arcs2.d_vars1.begin())),
		d_arcs2.d_sorcs.begin()
		);

	thrust::for_each(
		thrust::cuda::par.on(cs[0]),
		thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.begin(), d_arcs2.d_vals0.begin(), d_arcs2.d_sorcs.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.end(), d_arcs2.d_vals0.end(), d_arcs2.d_sorcs.end())),
		ModifyNodes(nodes, nodes_offset)
		);

	thrust::reduce_by_key(
		thrust::cuda::par.on(cs[0]),
		d_nodes.vars.begin(),
		d_nodes.vars.end(),
		d_nodes_set.begin(),
		d_vars_key.begin(),
		d_vars_size.begin()
		);

	//new_end = thrust::remove_if(
	//	thrust::cuda::par.on(cs[0]),
	//	thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.begin(), d_arcs.d_vals0.begin(), d_arcs.d_vars1.begin(), d_arcs.d_vals1.begin(), d_arcs.d_sorcs.begin(), d_arcs.d_cmaps.begin())),
	//	thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.end(), d_arcs.d_vals0.end(), d_arcs.d_vars1.end(), d_arcs.d_vals1.end(), d_arcs.d_sorcs.end(), d_arcs.d_cmaps.end())),
	//	is_conflict()
	//	);
	//printf("old_end = %d", d_arcs.d_vars0.size());
	//int new_end_length = d_arcs.d_vars0.end() - thrust::get<0>(new_end.get_iterator_tuple());
	//d_arcs.resize(new_end_length);
	//printf("new_end = %d", new_end_length);

	cudaStreamSynchronize(cs[0]);
	cudaStreamSynchronize(cs[1]);
	cudaStreamSynchronize(cs[2]);
	auto counter_has_zero = thrust::find(thrust::cuda::par.on(cs[2]), d_arcs2.d_sorcs.begin(), d_arcs2.d_sorcs.end(), 0);

	if (counter_has_zero == d_arcs2.d_sorcs.end())
	{
		std::cout << "ISAT" << std::endl;
		propagationEnable = false;
		return true;
	}

	auto domX_has_zero = thrust::find(thrust::cuda::par.on(cs[1]), d_vars_size.begin(), d_vars_size.end(), 0);

	if (domX_has_zero != d_vars_size.end())
	{
		std::cout << "IUNSAT!!" << std::endl;
		propagationEnable = false;
		return false;
	}

	cudaStreamSynchronize(cs[1]);
	cudaStreamSynchronize(cs[2]);

#pragma region 释放堆
	for (i = 0; i < CSCOUNT; ++i)
	{
		cudaStreamDestroy(cs[i]);
	}
#pragma endregion

	return 1;
}

extern "C" int AC4GpuPropagation()
{
	if (!propagationEnable)
	{
		return 0;
	}

	int i;
	int *d_vars_key_ptr;
	int *d_vars_size_ptr;
	int vs_size = d_vars_size.size();
	int *nodes = thrust::raw_pointer_cast(d_nodes_set.data());
	int *nodes_offset = thrust::raw_pointer_cast(d_node_global.data());
	int *counter_value = thrust::raw_pointer_cast(d_arcs2.d_sorcs.data());
	thrust::device_vector<int> d_vars_key(vs_size);
	thrust::device_vector<int>::iterator domX_has_zero, counter_has_zero;
	d_vars_size_ptr = thrust::raw_pointer_cast(d_vars_size.data());
	d_vars_key_ptr = thrust::raw_pointer_cast(d_vars_key.data());


	while (true)
	{
		//L7
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.begin(), d_arcs.d_vals0.begin(), d_arcs.d_vars1.begin(), d_arcs.d_vals1.begin(), d_arcs.d_sorcs.begin(), d_arcs.d_cmaps.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.end(), d_arcs.d_vals0.end(), d_arcs.d_vars1.end(), d_arcs.d_vals1.end(), d_arcs.d_sorcs.end(), d_arcs.d_cmaps.end())),
			ModifyArcs(nodes, nodes_offset, counter_value)
			);

		counter_has_zero = thrust::find(d_arcs2.d_sorcs.begin(), d_arcs2.d_sorcs.end(), (int)-1);

		if (counter_has_zero == d_arcs2.d_sorcs.end())
		{
			std::cout << "SAT" << std::endl;
			return true;
		}

		//L13
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.begin(), d_arcs2.d_vals0.begin(), d_arcs2.d_sorcs.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.end(), d_arcs2.d_vals0.end(), d_arcs2.d_sorcs.end())),
			ModifyNodesProp(nodes, nodes_offset, d_vars_size_ptr)
			);

		domX_has_zero = thrust::find(d_vars_size.begin(), d_vars_size.end(), (int)0);

		if (domX_has_zero != d_vars_size.end())
		{

			thrust::reduce_by_key(
				d_nodes.vars.begin(),
				d_nodes.vars.end(),
				d_nodes_set.begin(),
				d_vars_key.begin(),
				d_vars_size.begin()
				);

			domX_has_zero = thrust::find(d_vars_size.begin(), d_vars_size.end(), (int)0);

			if (domX_has_zero != d_vars_size.end())
			{

				std::cout << "UNSAT!!" << std::endl;
				return false;
			}
		}
	}
	return 1;
};