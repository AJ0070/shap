# C++ Migration Roadmap

This document tracks the migration of SHAP's compiled extensions from setuptools (raw C API, Cython, CUDA) to nanobind + CMake, with the end goal of switching the build backend to scikit-build-core.

## Current State

SHAP has four types of compiled code:

| Module | Technology | Location | Status |
|---|---|---|---|
| `shap.cutils` | nanobind + CMake | `shap/cutils/cutils.cpp` | **In progress** — njit functions being ported |
| `shap._cext` | Raw Python C API | `shap/cext/_cext.cc` | Not yet migrated |
| `shap._cext_gpu` | Raw Python C API + CUDA | `shap/cext/_cext_gpu.cc`, `_cext_gpu.cu` | Not yet migrated |
| `_kernel_lib` | Cython | `shap/explainers/_kernel_lib.pyx` | Not yet migrated |

Currently the build uses **two separate systems**:
- `setuptools` + `setup.py` for the old extensions (`_cext`, `_cext_gpu`, `_kernel_lib`)
- `CMakeLists.txt` + nanobind for the new `cutils` module (manual build step, not integrated into `pip install`)

## Migration Steps

1. **Port njit functions to nanobind** (current work on `move-some-njit-code-to-c` branch)
   - 15 `@njit` functions → C++ in `shap/cutils/cutils.cpp`
   - Built via CMake, see `make build_cutils`

2. **Port `_kernel_lib.pyx` (Cython) to nanobind**
   - Single function `_exp_val` — ~15 lines of loop logic
   - Add to `cutils.cpp` or a new nanobind module

3. **Port `_cext` to nanobind**
   - Functions: `dense_tree_shap`, `dense_tree_predict`, `dense_tree_update_weights`, `dense_tree_saabas`, `compute_expectations`
   - Heavy lifting is in `tree_shap.h` (header-only, can be reused)
   - Replace `PyArg_ParseTuple` / `PyArray_FROM_OTF` boilerplate with `nb::ndarray`

4. **Port `_cext_gpu` to nanobind**
   - Same approach as `_cext`, but with CUDA
   - CMake has native CUDA support via `enable_language(CUDA)`

5. **Switch build backend to scikit-build-core**
   - Replace `build-backend = "setuptools.build_meta"` with `build-backend = "scikit_build_core.build"` in `pyproject.toml`
   - Delete `setup.py`
   - Remove `setuptools`, `cython`, `numpy` from build requires
   - Migrate version management (`setuptools-scm` → `scikit_build_core.metadata.setuptools_scm` plugin)
   - All extensions built via `CMakeLists.txt` through `pip install`

## Building cutils (during migration)

Until the full migration is complete, the nanobind `cutils` module must be built manually:

```bash
make build_cutils
```

This runs CMake and copies the resulting `.so` into `shap/cutils/`.

## Relevant Links

- [nanobind docs](https://nanobind.readthedocs.io/en/latest/)
- [nanobind ndarray](https://nanobind.readthedocs.io/en/latest/ndarray.html)
- [nanobind + scikit-build-core build setup](https://nanobind.readthedocs.io/en/latest/building.html)
- [scikit-build-core configuration](https://scikit-build-core.readthedocs.io/en/latest/configuration.html)
