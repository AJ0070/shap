build_ext:
	rm -rf build/
	python setup.py build_ext --inplace

build_cutils:
	cmake -S . -B build/cutils -DPython_EXECUTABLE=$$(which python)
	cmake --build build/cutils
	cp build/cutils/cutils*.so shap/cutils/
	cp build/cutils/cutils*.so $$(python -c "import shap, os; print(os.path.join(os.path.dirname(shap.__file__), 'cutils'))")/ 2>/dev/null || true

build_cutils_debug:
	cmake -S . -B build/cutils_debug -DPython_EXECUTABLE=$$(which python) -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/cutils_debug
	cp build/cutils_debug/cutils*.so shap/cutils/
