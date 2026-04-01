build_ext:
	rm -rf build/
	python setup.py build_ext --inplace

build_cutils:
	cmake -S . -B build/cutils -DPython_EXECUTABLE=$$(which python)
	cmake --build build/cutils
	cp build/cutils/cutils*.so shap/cutils/
