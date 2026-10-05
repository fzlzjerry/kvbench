# Install and test targets for the kvbench release.

PYTHON ?= .venv/bin/python
DEPS ?= .deps
TEST_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(DEPS):src:.

UNIT_TESTS := \
	tests.schema.test_config_schema \
	tests.schema.test_method_admission \
	tests.schema.test_phase3_schema \
	tests.schema.test_phase6_schema \
	tests.schema.test_phase9_schema \
	tests.unit.test_allocation_attribution \
	tests.unit.test_gqa_device_dispatch \
	tests.unit.test_gqa_taxonomy \
	tests.unit.test_model_loader_receipt \
	tests.unit.test_phase11_kvquant_adapter \
	tests.unit.test_phase11_kvquant_cache \
	tests.unit.test_phase11_kvquant_factory \
	tests.unit.test_phase11_kvquant_fixture \
	tests.unit.test_phase11_kvquant_session \
	tests.unit.test_phase12_schema \
	tests.unit.test_phase3_allocator_controls \
	tests.unit.test_phase3_gqa_device_dispatch_geometry \
	tests.unit.test_phase3_runtime \
	tests.unit.test_phase4_adapter \
	tests.unit.test_phase6_turboquant_adapter \
	tests.unit.test_phase6_turboquant_fixture \
	tests.unit.test_phase8_kivi_adapter \
	tests.unit.test_phase8_kivi_allocation \
	tests.unit.test_phase8_kivi_cache \
	tests.unit.test_phase8_kivi_fixture \
	tests.unit.test_phase8_kivi_schema \
	tests.unit.test_phase8_kivi_session \
	tests.unit.test_phase8_process_supervision \
	tests.unit.test_phase8_ratio_terminology \
	tests.unit.test_phase9_calibration \
	tests.unit.test_process_supervision

.PHONY: install test test-cuda test-graph

# Hash-pinned environment matching docker/measurement.Dockerfile.
install:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --require-hashes -r preflight/requirements-e00.txt
	$(PYTHON) -m pip install --require-hashes --no-deps --only-binary=:all: --target $(DEPS) -r preflight/requirements-phase3.txt
	$(PYTHON) -m pip install --no-deps -e .

# CPU unit and schema tests; GPUs are hidden.
test:
	CUDA_VISIBLE_DEVICES= $(TEST_ENV) $(PYTHON) -m unittest $(UNIT_TESTS) -v

# CUDA kernel, adapter, and allocation tests (requires the GPU).
test-cuda:
	$(TEST_ENV) $(PYTHON) -m unittest discover -s tests/cuda -p 'test_*.py' -v

# CUDA Graph capture and replay tests (requires the GPU).
test-graph:
	$(TEST_ENV) $(PYTHON) -m unittest discover -s tests/graph -p 'test_*.py' -v
