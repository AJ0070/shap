// see https://nanobind.readthedocs.io/en/latest/basics.html#basics and following docs
#include <cassert>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <iostream>
#include <tuple>

namespace nb = nanobind;
using namespace nb::literals;

// @njit
// def _compute_grey_code_row_values(
//     row_values: npt.NDArray[Any],
//     mask: npt.NDArray[np.bool_],
//     inds: npt.NDArray[np.intp],
//     outputs: npt.NDArray[Any],
//     shapley_coeff: npt.NDArray[Any],
//     extended_delta_indexes: npt.NDArray[np.intp],
//     noop_code: int,
// ) -> None:
//     set_size = 0
//     M = len(inds)
//     for i in range(2**M):
//         # update the mask
//         delta_ind = extended_delta_indexes[i]
//         if delta_ind != noop_code:
//             mask[delta_ind] = ~mask[delta_ind]
//             if mask[delta_ind]:
//                 set_size += 1
//             else:
//                 set_size -= 1
//
//         # update the output row values
//         on_coeff = shapley_coeff[set_size - 1]
//         if set_size < M:
//             off_coeff = shapley_coeff[set_size]
//         out = outputs[i]
//         for j in inds:
//             if mask[j]:
//                 row_values[j] += out * on_coeff
//             else:
//                 row_values[j] -= out * off_coeff
//
// row_values = np.array([[0., 0.],
//                        [0., 0.]])
// mask = np.array([False, False])
// inds = np.array([0, 1])
// outputs = np.array([[0.4996641 , 0.5003359 ],
//                     [0.44872311, 0.55127689],
//                     [0.0059607 , 0.9940393 ],
//                     [0.08355695, 0.91644305]])
// shapley_coeff = np.array([0.5, 0.5])
// extended_delta_indexes = np.array([2147483647, 1, 0, 1])
// noop_code = 2147483647
//
//
//
//(Pdb++) row_values
// array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
// (Pdb++) row_values.dtype
// dtype('float64')
// (Pdb++) mask.dtype
// dtype('bool')
// (Pdb++) inds.dtype
// dtype('int64')
// (Pdb++) outputs.dtype
// dtype('float64')
// (Pdb++) coeff.dtype
// dtype('float64')
// (Pdb++) extended_delta_indexes.dtype
// dtype('int64')
// (Pdb++) MaskedModel.delta_mask_noop_value.dtype
// *** AttributeError: 'int' object has no attribute 'dtype'
// (Pdb++) MaskedModel.delta_mask_noop_value.dtype
// *** AttributeError: 'int' object has no attribute 'dtype'
// (Pdb++) MaskedModel.delta_mask_noop_value
// 2147483647
//
//  TypeError: _compute_grey_code_row_values(): incompatible function arguments. The following argument types are supported:
//    1. _compute_grey_code_row_values(
//    arg0: ndarray[dtype=float64, shape=(*, *), device='cpu'],
//    arg1: ndarray[dtype=bool, shape=(*), device='cpu'],
//    arg2: ndarray[dtype=uint64, shape=(*), device='cpu'],
//    arg3: ndarray[dtype=float64, shape=(*, *), device='cpu'],
//    arg4: ndarray[dtype=float64, shape=(*), device='cpu'],
//    arg5: ndarray[dtype=uint64, shape=(*), device='cpu'],
//    arg6: int, /) -> None


void compute_grey_code_row_values_2d(
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
	assert(row_values.shape(0) == mask.shape(0));
	// assert(row_values.shape(0) == mask.shape(0));
	size_t set_size = 0;
	size_t shapley_idx = 0;
	int M = inds.shape(0);
	auto rv = row_values.view();
	int delta_ind;
	float on_coeff;
	float off_coeff = shapley_coeff(0);
	float multiplication_factor;
	for (size_t i=0; i<pow(2, M); i++) {
                assert(i < extended_delta_indexes.shape(0));
		assert(i < outputs.shape(0));

		// std::cout << "Outer loop i: " << i << std::endl;
		delta_ind = extended_delta_indexes(i);
		if (delta_ind != noop_code) {
			assert((delta_ind < mask.shape(0)) && (delta_ind >= 0));
			mask(delta_ind) = !mask(delta_ind);
			if (mask(delta_ind)) {
				// std::cout << "if (mask(delta_ind))" << std::endl;
				set_size += 1;
			}
			else {
				set_size -= 1;
			}
		}
		if (set_size == 0) {

			shapley_idx = shapley_coeff.shape(0) - 1;
		}
		else {
			shapley_idx = set_size - 1;
		}
	        assert((shapley_idx < shapley_coeff.shape(0)) && (shapley_idx >= 0));
		on_coeff = shapley_coeff(shapley_idx);
		if (set_size < (size_t)M) {
			off_coeff = shapley_coeff(shapley_idx);
		}
		// assume inds.shape(0) == row_values.shape(0). Probably better to assert
		for (size_t rvi = 0; rvi < rv.shape(0); rvi++) {
			assert (rvi < inds.shape(0));
			assert (inds(rvi) < mask.shape(0));
			if (mask(inds(rvi))) {
				multiplication_factor = on_coeff;
			}
			else {
				multiplication_factor = -off_coeff;
			}
			for (size_t rvj = 0; rvj < rv.shape(1); rvj++) {
				assert (i < outputs.shape(0));
				assert (rvj < outputs.shape(1));
				assert (rvi < rv.shape(0));
				assert (rvj < rv.shape(1));
				rv(rvi, rvj) += multiplication_factor * outputs(i, rvj);
			}
		}
        }
}

void compute_grey_code_row_values_1d(
    nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
	assert(row_values.shape(0) == mask.shape(0));
	// assert(row_values.shape(0) == mask.shape(0));
	size_t set_size = 0;
	size_t shapley_idx = 0;
	int M = inds.shape(0);
	auto rv = row_values.view();
	int delta_ind;
	float on_coeff;
	float off_coeff = shapley_coeff(0);
	float multiplication_factor;
	for (size_t i=0; i<pow(2, M); i++) {
                assert(i < extended_delta_indexes.shape(0));
		assert(i < outputs.shape(0));

		// std::cout << "Outer loop i: " << i << std::endl;
		delta_ind = extended_delta_indexes(i);
		if (delta_ind != noop_code) {
			assert((delta_ind < mask.shape(0)) && (delta_ind >= 0));
			mask(delta_ind) = !mask(delta_ind);
			if (mask(delta_ind)) {
				// std::cout << "if (mask(delta_ind))" << std::endl;
				set_size += 1;
			}
			else {
				set_size -= 1;
			}
		}
	        // std::cout << "Set size: " << set_size << std::endl;
		if (set_size == 0) {

			shapley_idx = shapley_coeff.shape(0) - 1;
		}
		else {
			shapley_idx = set_size - 1;
		}
	        assert((shapley_idx < shapley_coeff.shape(0)) && (shapley_idx >= 0));
		on_coeff = shapley_coeff(shapley_idx);
		if (set_size < (size_t)M) {
			off_coeff = shapley_coeff(shapley_idx);
		}
		// assume inds.shape(0) == row_values.shape(0). Probably better to assert
		for (size_t rvi = 0; rvi < rv.shape(0); rvi++) {
			assert (rvi < inds.shape(0));
			assert (inds(rvi) < mask.shape(0));
			if (mask(inds(rvi))) {
				multiplication_factor = on_coeff;
			}
			else {
				multiplication_factor = -off_coeff;
			}
			assert (i < outputs.shape(0));
			assert (rvi < rv.shape(0));
		        rv(rvi) += multiplication_factor * outputs(i);
		}
        }
}

// @cython.boundscheck(False)
// @cython.wraparound(False)
// def _exp_val(int nsamples_run,
//              int nsamples_added,
//              int D,
//              int N,
//              double[::1] weights,
//              double[:,:] y,
//              double[:,:] ey):
//
//     cdef:
//         double[::1] ref = np.zeros(D)
//         double[::1] eyVal = np.zeros(D)
//         int i, j, k
//
//     for i in range(nsamples_added):
//         if i < nsamples_run:
//             continue
//         eyVal[:] = ref
//         for j in range(N):
//             for k in range(D):
//                 eyVal[k] += y[i * N + j, k] * weights[j]
//
//         ey[i, :] = eyVal
//         nsamples_run += 1
//     return ey, nsamples_run
//

// std::tuple<nb::ndarray<double, nb::ndim<1>, nb::c_contig>, nb::ndarray<double, nb::ndim<1>, nb::c_contig>> _exp_val(
//     const int nsamples_run,
//     const int nsamples_added,
//     const int D,
//     const nb::ndarray<double, nb::ndim<1>, nb::c_contig>,
//     const nb::ndarray<double, nb::ndim<2>>,
//     const nb::ndarray<double, nb::ndim<2>>,
// ) {
//     std::vector<double> ref(D, 0.0);
//     std::vector<double eyVal(D, 0.0);
//     for (size_t i = 0; i < nsamples_added; i++) {
// 	    if (i < nsamples_run) {
// 		    continue;
// 	    }
// 	    std::memcpy(eyVal, std::vector<double> tmp(D, ref), sizeof array);
//     }
// }

NB_MODULE(cutils, m) {
    m.def("_compute_grey_code_row_values", &compute_grey_code_row_values_1d);
    m.def("_compute_grey_code_row_values", &compute_grey_code_row_values_2d);
}
