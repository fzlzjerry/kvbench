SHELL := /usr/bin/bash
.SHELLFLAGS := --noprofile --norc -eu -o pipefail -c

unexport BASH_ENV
unexport ENV
unexport LD_LIBRARY_PATH
unexport LD_PRELOAD
unexport PYTHONHOME
unexport PYTHONPATH

.PHONY: preflight preflight-unit

PHASE2_PYTHON := /usr/bin/python3
PHASE2_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src
PHASE2_VALIDATE := $(PHASE2_ENV) $(PHASE2_PYTHON) scripts/validate_phase2.py
PHASE2_CLI := $(PHASE2_ENV) $(PHASE2_PYTHON) -m kvbench
PHASE3_PYTHON := .venv/bin/python
PHASE3_SITE := $(CURDIR)/.phase3/site-packages
PHASE3_ENV := /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(PHASE3_SITE):src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
PHASE3_CLI := $(PHASE3_ENV) $(PHASE3_PYTHON) -m kvbench
PHASE3_REMEDIATION_UNIT_TESTS := tests.unit.test_allocation_attribution tests.unit.test_gqa_device_dispatch tests.unit.test_gqa_taxonomy tests.unit.test_process_supervision
TURBOQUANT_REFERENCE_VENV := .reference/turboquant-v0.25.1
TURBOQUANT_REFERENCE_PYTHON := $(TURBOQUANT_REFERENCE_VENV)/bin/python
TURBOQUANT_REFERENCE_SOURCE := .reference/vllm-source-v0.25.1
TURBOQUANT_REFERENCE_ENV := /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(CURDIR)/src:$(CURDIR) CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDA_ARCH_LIST=12.0 CUDAARCHS=120 CMAKE_CUDA_ARCHITECTURES=120 VLLM_NO_USAGE_STATS=1 HF_HUB_DISABLE_TELEMETRY=1
MEASUREMENT_IMAGE ?= kvbench-measurement:phase6a
MEASUREMENT_IMAGE_CONFIG_DIGEST ?=
MEASUREMENT_BASE_IMAGE_DIGEST := sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c
MEASUREMENT_GPU_UUID := GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b
PHASE6_AUTHORIZED_IMAGE_CONFIG_DIGEST := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
PHASE6_R2_OUTER_RUN_ID ?=
PHASE6_R2_OUTER_ARTIFACT ?=
override PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
override PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST := sha256:f27e4cdef6bd15f18ab76b1fe0e4413ede004b42538c74e3dd90d04172406f75
override PHASE8_KIVI_REFERENCE_CONFIG_DIGEST := sha256:0915dc8488fd6c9a150a3b4f56bb4b97b5dbdb7c51d96cda2d431df20e856ce3
override PHASE8_KIVI_EXTENSION_SHA256 := 45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9
override PHASE8_KIVI_NEW_PACK_SHA256 := 3678af0e34a0ba18e5d80a4128acf11d4070667c800a15540a16d07253a4f75e
PHASE8_R2_INNER_ARTIFACT ?=
PHASE8_KIVI_ADMISSION_ARTIFACT := $(CURDIR)/artifacts/phase8/phase8-20260727t113020276z-462325e9-0edc5a-k4v4-fixed-l128-eager
PHASE8_R2_OUTER_RUN_ID ?=
PHASE8_R2_OUTER_ARTIFACT ?=
PHASE8_R2_OUTER_RECEIPT := docs/evidence/phase8/r2-admission-outer-publication.json
R2_ARTIFACT := /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(CURDIR):$(CURDIR)/src $(PHASE2_PYTHON) scripts/r2_artifact.py
KIVI_B019_DEVICE ?= cpu
KIVI_B019_SOURCE_ROOT ?=
KVQUANT_GQA_SOURCE_ROOT ?=
KVQUANT_CALIBRATION_SOURCE_ROOT ?= /home/rockrock/third_party_worktrees/kvquant-gqa
KVQUANT_CALIBRATION_DATASET_PARQUET ?= /tmp/kvbench-phase9-inputs/wikitext-2-raw-v1-train-0000.parquet
KVQUANT_CALIBRATION_MODEL_CACHE ?= /root/.cache/huggingface/hub
KVQUANT_CALIBRATION_ARTIFACT ?=
PHASE9_CALIBRATION := /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(CURDIR):$(CURDIR)/src $(PHASE2_PYTHON) scripts/phase9_kvquant_calibration.py
KVQUANT_REFERENCE_BASE_IMAGE := sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d
KVQUANT_REFERENCE_IMAGE := sha256:24eb3f6ff39b72f45c353acfbef6ce2d9aaac0860180b4dde8b937593176714b
KVQUANT_REFERENCE_IMAGE_TAG := kvbench-reference-kvquant:phase10
KVQUANT_REFERENCE_TOKENIZERS_WHEEL := /tmp/phase9p-tokenizers-wheelhouse/tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
KVQUANT_REFERENCE_TOKENIZERS_SHA256 := 9e0480c452217edd35eca56fafe2029fb4d368b7c0475f8dfa3c5c9c400a7456
KVQUANT_REFERENCE_SOURCE_ROOT ?= /home/rockrock/third_party_worktrees/kvquant-gqa
KVQUANT_REFERENCE_BUILD_ROOT ?= /home/rockrock/third_party_worktrees/kvquant-gqa-reference-build
KVQUANT_REFERENCE_EXTENSION := $(KVQUANT_REFERENCE_BUILD_ROOT)/quant_cuda.cpython-312-x86_64-linux-gnu.so
KVQUANT_REFERENCE_EXTENSION_SHA256 := 53bee7b4b5a0dead6adb682df1343330963b41149d12c2a876888c1c2ede9597
KVQUANT_REFERENCE_CALIBRATION := $(CURDIR)/calibration/kvquant/kvqcal-cdb724c806d64d095c040d2673a987a3
KVQUANT_REFERENCE_PATCH_MANIFEST := $(CURDIR)/third_party/patches/kvquant/manifest.json
KVQUANT_GRAPHSAFE_SOURCE_ROOT ?= /home/rockrock/third_party_worktrees/kvquant-gqa
override KVQUANT_GRAPHSAFE_COMMIT := 0d9df350bd1788284e1ce76a8bf6e886beca5efa
override KVQUANT_GRAPHSAFE_TREE := a85cf7bf093982a4bf89c33d4e6794d9a85f846d
KVQUANT_PHASE11PR_FIXTURES := $(CURDIR)/reference/kvquant_phase11pr/fixtures
override PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
override PHASE11_KVQUANT_CORRECTED_COMMIT := 4b8533b29b04f8c4bf55f688a41fefe20487637b
override PHASE11_KVQUANT_CORRECTED_TREE := 46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b
override PHASE11_KVQUANT_AGGREGATE_PATCH_SHA256 := bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6
override PHASE11_KVQUANT_EXTENSION_SHA256 := a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1
override PHASE11_KVQUANT_CALIBRATION_ROOT_SHA256 := 8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf
override PHASE11_KVQUANT_CALIBRATION := $(CURDIR)/calibration/kvquant/kvqcal-cdb724c806d64d095c040d2673a987a3
PHASE11_KVQUANT_INNER_ARTIFACT ?=
PHASE11_KVQUANT_OUTER_ARTIFACT ?=
override PHASE11_KVQUANT_METHOD_ADMISSION_REPORT := docs/evidence/phase11/kvquant-method-admission.json
override PHASE11_KVQUANT_METHOD_ADMISSION_CHECKSUM := docs/evidence/phase11/kvquant-method-admission.sha256
override PHASE11_KVQUANT_PUBLICATION_RECEIPT := docs/evidence/phase11/r2-admission-outer-publication.json
PHASE11RQ23_KVQUANT_INNER_ARTIFACT ?=
PHASE11RQ23_KVQUANT_OUTER_ARTIFACT ?=
override PHASE11RQ23_KVQUANT_METHOD_ADMISSION_REPORT := docs/evidence/phase11rq23/kvquant-method-admission.json
override PHASE11RQ23_KVQUANT_METHOD_ADMISSION_CHECKSUM := docs/evidence/phase11rq23/kvquant-method-admission.sha256
override PHASE11RQ23_KVQUANT_PUBLICATION_RECEIPT := docs/evidence/phase11rq23/r2-admission-outer-publication.json
override PHASE11_KVQUANT_ACTIVE_INNER_ARTIFACT := $(PHASE11_KVQUANT_INNER_ARTIFACT)
override PHASE11_KVQUANT_ACTIVE_OUTER_ARTIFACT := $(PHASE11_KVQUANT_OUTER_ARTIFACT)
override PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_REPORT := $(PHASE11_KVQUANT_METHOD_ADMISSION_REPORT)
override PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_CHECKSUM := $(PHASE11_KVQUANT_METHOD_ADMISSION_CHECKSUM)
override PHASE11_KVQUANT_ACTIVE_PUBLICATION_RECEIPT := $(PHASE11_KVQUANT_PUBLICATION_RECEIPT)
override PHASE11_KVQUANT_ACTIVE_SOURCE_ROOT := /home/rockrock/third_party_worktrees/kvquant-gqa
override PHASE11_KVQUANT_ACTIVE_COMMIT := $(PHASE11_KVQUANT_CORRECTED_COMMIT)
override PHASE11_KVQUANT_ACTIVE_TREE := $(PHASE11_KVQUANT_CORRECTED_TREE)
override PHASE11_KVQUANT_ACTIVE_PATCH_SHA256 := $(PHASE11_KVQUANT_AGGREGATE_PATCH_SHA256)
override PHASE11_KVQUANT_ACTIVE_EXTENSION_SHA256 := $(PHASE11_KVQUANT_EXTENSION_SHA256)
override PHASE11_KVQUANT_ACTIVE_VALIDATOR_PATH := scripts/validate_kvquant_long_context_patch.py
override PHASE11_KVQUANT_ACTIVE_VALIDATOR_MODULE := scripts.validate_kvquant_long_context_patch
override PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE := decision0027
override PHASE11_KVQUANT_ACTIVE_LAUNCH_PREFIX := phase11-launch
override PHASE11RQ23_Q23_EVIDENCE := $(CURDIR)/artifacts/phase11/phase11dq23-launch.ssAO5M
override PHASE11D_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
override PHASE11D_KVQUANT_COMMIT := 4b8533b29b04f8c4bf55f688a41fefe20487637b
override PHASE11D_KVQUANT_TREE := 46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b
override PHASE11D_KVQUANT_PATCH_SHA256 := bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6
override PHASE11D_KVQUANT_EXTENSION_SHA256 := a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1
PHASE11D_KVQUANT_SOURCE_ROOT ?= /home/rockrock/third_party_worktrees/kvquant-gqa
override PHASE11DQ23_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
override PHASE11DQ23_KVQUANT_COMMIT := 34b0bdfa83082e1f30387d9ac5cca369006e089c
override PHASE11DQ23_KVQUANT_TREE := 1f85af65fe03061583ffe8bd91e47d7ecffdd312
override PHASE11DQ23_KVQUANT_PATCH_SHA256 := 7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a
override PHASE11DQ23_KVQUANT_EXTENSION_SHA256 := b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d
PHASE11DQ23_KVQUANT_SOURCE_ROOT ?= /home/rockrock/third_party_worktrees/kvquant-gqa
override PHASE12_AUTHORIZED_IMAGE_CONFIG_DIGEST := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
PHASE12_CAMPAIGN_ARTIFACT ?=
PHASE12_HOST_ENV := /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(PHASE3_SITE):$(CURDIR):$(CURDIR)/src
PHASE12_HOST_PYTHON := $(PHASE12_HOST_ENV) $(CURDIR)/$(PHASE3_PYTHON)
override PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST := $(PHASE12_AUTHORIZED_IMAGE_CONFIG_DIGEST)
override PHASE13_ANALYSIS_IMAGE_CONFIG_DIGEST := sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d
override PHASE13B_LOCAL_BUNDLE := $(CURDIR)/artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission
override PHASE13B_LOCAL_ROOT_SHA256 := f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe
override PHASE13RQ4_LOCAL_BUNDLE := $(CURDIR)/artifacts/phase13rq4/phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd
override PHASE13RQ4_LOCAL_ROOT_SHA256 := 9f027d64424844d0d62311daad5740e2b76960d1d5e7b8be26aa1ccba100a8db
override PHASE13_PREFIX_SEED_ROOT := $(CURDIR)/artifacts/phase13_prefix_catalogs/phase13-prefix-seed-20260822t185619z-b27442c4-168
PHASE13_CAMPAIGN_ARTIFACT ?=
override PHASE13D_SOURCE_CAMPAIGN := $(CURDIR)/artifacts/phase13/phase13-20260822t150835736582z-4ddd7b17-3a8fb3
override PHASE13D_SOURCE_ROOT_SHA256 := feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531
PHASE13D_CAMPAIGN_ARTIFACT ?=
PHASE13_HOST_PYTHON := $(PHASE12_HOST_PYTHON)
PHASE13F_ARTIFACT ?=
PHASE13T_ARTIFACT ?=
KIVI_REFERENCE_IMAGE := kvbench-reference-kivi:phase7
KIVI_REFERENCE_PARENT_CONFIG := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
KIVI_REFERENCE_IMAGE_MANIFEST := sha256:f27e4cdef6bd15f18ab76b1fe0e4413ede004b42538c74e3dd90d04172406f75
KIVI_REFERENCE_IMAGE_CONFIG := sha256:0915dc8488fd6c9a150a3b4f56bb4b97b5dbdb7c51d96cda2d431df20e856ce3
KIVI_REFERENCE_EXTENSION_SHA256 := 45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9
KIVI_REFERENCE_BUILD_REVISION := 3417ea0e7f322369eed21bb787a9a9a19b0a69bd

.PHONY: bootstrap bootstrap-phase3 test checks format-check lint hot-path-check typecheck config-check
.PHONY: provenance-check scope-check immutable-check package-lock-check
.PHONY: phase3-package-lock-check test-cuda test-graph test-allocation
.PHONY: smoke pilot full-scan validate-full-scan test-phase16 profile-subset fit validate-phase17-modeling test-phase17 reproduce package-phase18 validate-phase18 test-phase18
.PHONY: test-phase13f remediate-phase13-feasibility validate-phase13-feasibility
.PHONY: test-phase13t validate-phase13-timeout
.PHONY: densify-pilot-knees validate-pilot-densification
.PHONY: fit figures reproduce
.PHONY: reference-turboquant validate-reference-turboquant
.PHONY: measurement-container observe-measurement-container-lock
.PHONY: measurement-container-lock-review verify-measurement-container
.PHONY: preflight-container phase6a-bf16-container-parity
.PHONY: publish-artifact-r2 verify-artifact-r2
.PHONY: phase6a-source-safety admit-turboquant validate-admission-turboquant
.PHONY: remediate-b018-turboquant
.PHONY: phase6-r2-outer-bundle validate-phase6-r2-outer-bundle
.PHONY: admit-kivi validate-admission-kivi
.PHONY: phase8-r2-outer-bundle validate-phase8-r2-outer-bundle
.PHONY: validate-phase8-r2-outer-publication
.PHONY: validate-kivi-b019-patch
.PHONY: validate-kvquant-gqa-patch
.PHONY: validate-kvquant-graphsafe-patch
.PHONY: validate-kvquant-long-context-patch validate-kvquant-phase11d
.PHONY: validate-kvquant-q23-long-context-patch validate-kvquant-phase11dq23
.PHONY: calibrate-kvquant validate-calibration-kvquant reference-kivi validate-reference-kivi
.PHONY: reference-kvquant validate-reference-kvquant
.PHONY: validate-reference-kvquant-phase11pr
.PHONY: admit-kvquant validate-admission-kvquant
.PHONY: admit-kvquant-q23 validate-admission-kvquant-q23
.PHONY: validate-phase12e-kivi-history
.PHONY: unified-admission validate-unified-admission
.PHONY: pilot validate-pilot

preflight:
	@/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC \
		/usr/bin/bash --noprofile --norc scripts/preflight.sh

preflight-unit:
	@PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests/unit -p 'test_preflight_unit.py' -v

bootstrap:
	@test -x $(PHASE2_PYTHON)
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -c 'import sys; assert sys.version_info >= (3, 11); import kvbench'
	@$(PHASE2_VALIDATE) package-lock
	@echo '{"status":"ready","action":"verified_only","installed":false}'

bootstrap-phase3:
	@$(PHASE3_ENV) $(PHASE3_PYTHON) scripts/bootstrap_phase3.py verify

format-check:
	@$(PHASE2_VALIDATE) format

lint:
	@$(PHASE2_VALIDATE) lint

hot-path-check:
	@$(PHASE2_VALIDATE) hot-path

typecheck:
	@$(PHASE2_VALIDATE) annotations
	@$(PHASE2_VALIDATE) phase3-annotations

config-check:
	@$(PHASE2_VALIDATE) configs

provenance-check:
	@$(PHASE2_VALIDATE) provenance

scope-check:
	@$(PHASE2_VALIDATE) scope

immutable-check:
	@$(PHASE2_VALIDATE) immutable

package-lock-check:
	@$(PHASE2_VALIDATE) package-lock

phase3-package-lock-check:
	@$(PHASE2_VALIDATE) phase3-package-lock

checks: format-check lint hot-path-check typecheck config-check provenance-check scope-check immutable-check package-lock-check phase3-package-lock-check

test: checks
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m unittest discover -s tests/schema -p 'test_*.py' -v
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m unittest discover -s tests/unit -p 'test_phase2_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase3_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest $(PHASE3_REMEDIATION_UNIT_TESTS) -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase4_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase5_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_phase6_r2_outer_bundle -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_phase7_kivi_source_audit -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_phase7_kivi_b019_remediation -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_phase7_kivi_reference -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase8_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase9p_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase9_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase10_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase11pr_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase12_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase12e_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase13_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_measurement_container tests.unit.test_phase6a_bf16_parity tests.unit.test_phase6a_governance tests.unit.test_preflight_unit tests.unit.test_r2_artifact -v
	@$(PHASE2_VALIDATE) immutable

validate-phase12e-kivi-history:
	@$(PHASE3_ENV) KVBENCH_PHASE12E_REQUIRE_LOCAL_EVIDENCE=1 $(PHASE3_PYTHON) -m unittest tests.unit.test_phase12e_kivi_historical_authority -v

test-cuda: phase3-package-lock-check immutable-check
	@set +e; $(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/cuda -p 'test_phase*_*.py' -v; test_status=$$?; $(PHASE2_VALIDATE) immutable; immutable_status=$$?; if (( test_status != 0 )); then exit $$test_status; fi; exit $$immutable_status

test-graph: phase3-package-lock-check immutable-check
	@set +e; $(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/graph -p 'test_phase*_*.py' -v; test_status=$$?; $(PHASE2_VALIDATE) immutable; immutable_status=$$?; if (( test_status != 0 )); then exit $$test_status; fi; exit $$immutable_status

test-allocation: test-cuda

reference-turboquant:
	@$(PHASE2_PYTHON) reference/turboquant/bootstrap_environment.py prepare --venv $(TURBOQUANT_REFERENCE_VENV) --source $(TURBOQUANT_REFERENCE_SOURCE)
	@$(TURBOQUANT_REFERENCE_ENV) $(TURBOQUANT_REFERENCE_PYTHON) reference/turboquant/generate_fixtures.py --venv $(TURBOQUANT_REFERENCE_VENV) --source $(TURBOQUANT_REFERENCE_SOURCE)
	@$(TURBOQUANT_REFERENCE_ENV) $(PHASE2_PYTHON) reference/turboquant/validate_fixtures.py

validate-reference-turboquant:
	@$(TURBOQUANT_REFERENCE_ENV) $(PHASE2_PYTHON) reference/turboquant/validate_fixtures.py

validate-kivi-b019-patch:
	@$(PHASE3_ENV) $(PHASE3_PYTHON) scripts/validate_kivi_b019_patch.py --device "$(KIVI_B019_DEVICE)" $(if $(strip $(KIVI_B019_SOURCE_ROOT)),--source-root "$(KIVI_B019_SOURCE_ROOT)")

validate-kvquant-gqa-patch:
	@$(PHASE2_ENV) $(PHASE2_PYTHON) scripts/validate_kvquant_gqa_patch.py $(if $(strip $(KVQUANT_GQA_SOURCE_ROOT)),--source-root "$(KVQUANT_GQA_SOURCE_ROOT)")

validate-kvquant-graphsafe-patch:
	@task_root="$$(mktemp -d /tmp/kvbench-kvquant-graphsafe-validation.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		git clone --quiet --no-local --no-checkout "$(KVQUANT_GRAPHSAFE_SOURCE_ROOT)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$(KVQUANT_GRAPHSAFE_COMMIT)"; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$(KVQUANT_GRAPHSAFE_COMMIT)"; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD^{tree})" = "$(KVQUANT_GRAPHSAFE_TREE)"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		$(PHASE2_ENV) $(PHASE2_PYTHON) -m scripts.validate_kvquant_graphsafe_patch --source-root "$$task_root/source"

validate-kvquant-long-context-patch:
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m scripts.validate_kvquant_long_context_patch --source-root "$(PHASE11D_KVQUANT_SOURCE_ROOT)"

validate-kvquant-phase11d: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE11D_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)
validate-kvquant-phase11d: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE11D_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase11d_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"clean_phase11d_worktree_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase11d.XXXXXX)"; \
		cid=""; \
		trap 'status=$$?; if test -n "$$cid"; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; rm -rf -- "$$task_root"; exit $$status' EXIT; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$(git rev-parse HEAD)"; \
		git -C "$$task_root/repository" remote remove origin; \
		for fixture_root in \
			"$$task_root/repository/reference/kvquant/fixtures" \
			"$$task_root/repository/reference/kvquant_phase11pr/fixtures"; do \
			test -d "$$fixture_root"; \
			find "$$fixture_root" -type d -exec chmod 0555 {} +; \
			find "$$fixture_root" -type f -exec chmod 0444 {} +; \
		done; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env"; \
		git clone --quiet --no-local --no-checkout "$(PHASE11D_KVQUANT_SOURCE_ROOT)" "$$task_root/kvquant-source"; \
		git -C "$$task_root/kvquant-source" checkout --quiet --detach "$(PHASE11D_KVQUANT_COMMIT)"; \
		git -C "$$task_root/kvquant-source" remote remove origin; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD)" = "$(PHASE11D_KVQUANT_COMMIT)"; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD^{tree})" = "$(PHASE11D_KVQUANT_TREE)"; \
		test -z "$$(git -C "$$task_root/kvquant-source" status --porcelain=v1 --untracked-files=all)"; \
		(cd "$$task_root/repository" && \
			PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$$task_root/repository/src:$$task_root/repository" \
				/usr/bin/python3 -m scripts.validate_kvquant_long_context_patch \
					--source-root "$$task_root/kvquant-source" \
					>"$$task_root/source-validation.json"); \
		test "$$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["aggregate_patch_sha256"])' "$$task_root/source-validation.json")" = "$(PHASE11D_KVQUANT_PATCH_SHA256)"; \
		mkdir "$$task_root/kvquant-build"; \
		for file in setup_cuda.py quant_cuda.cpp quant_cuda_kernel.cu measurement_cuda_kernel.cu; do \
			source="$$task_root/kvquant-source/deployment/kvquant/$$file"; \
			test -f "$$source" && test ! -L "$$source"; \
			cp -- "$$source" "$$task_root/kvquant-build/$$file"; \
		done; \
		mkdir -p "$(CURDIR)/artifacts/phase11"; \
		artifact_root="$$(mktemp -d "$(CURDIR)/artifacts/phase11/phase11d-launch.XXXXXX")"; \
		test -z "$$(find "$$artifact_root" -mindepth 1 -maxdepth 1 -print -quit)"; \
		cid="$$(docker create --read-only --network=none \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-source,dst=/opt/kvquant-source,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-build,dst=/opt/kvquant-build" \
			--mount "type=bind,src=$$artifact_root,dst=/opt/phase11d-evidence" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TORCH_CUDA_ARCH_LIST=12.0+PTX \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$(PHASE11D_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /usr/bin/bash "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" \
			--noprofile --norc -eu -o pipefail -c \
			'cd /opt/kvquant-build && /opt/kvbench/.venv/bin/python setup_cuda.py build_ext --inplace && extension="$$(find /opt/kvquant-build -maxdepth 1 -type f -name "quant_cuda.*.so" -print -quit)" && test -n "$$extension" && /usr/bin/strip --strip-unneeded "$$extension" && test "$$(sha256sum "$$extension" | cut -d " " -f 1)" = "$(PHASE11D_KVQUANT_EXTENSION_SHA256)" && cd /home/rockrock/cmu_paper && /opt/kvbench/.venv/bin/python scripts/phase11d_kvquant_validation.py --repository-root /home/rockrock/cmu_paper --source-root /opt/kvquant-source --extension "$$extension" --fixture-root /home/rockrock/cmu_paper/reference/kvquant_phase11pr/fixtures --output-root /opt/phase11d-evidence' \
		)"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; \
		cid=""; \
		test -f "$$artifact_root/COMPLETE"; \
		(cd "$$artifact_root" && sha256sum --check --strict checksums.sha256 >/dev/null); \
		printf 'PHASE11D_ARTIFACT=%s\n' "$$artifact_root"; \
		printf 'PHASE11D_EVIDENCE_ROOT=%s\n' "$$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checksum_ledger_sha256"])' "$$artifact_root/COMPLETE")"

validate-kvquant-q23-long-context-patch:
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m scripts.validate_kvquant_q23_long_context_patch --source-root "$(PHASE11DQ23_KVQUANT_SOURCE_ROOT)"

validate-kvquant-phase11dq23: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE11DQ23_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)
validate-kvquant-phase11dq23: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE11DQ23_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase11dq23_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"clean_phase11dq23_worktree_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase11dq23.XXXXXX)"; \
		cid=""; \
		trap 'status=$$?; if test -n "$$cid"; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; rm -rf -- "$$task_root"; exit $$status' EXIT; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$(git rev-parse HEAD)"; \
		git -C "$$task_root/repository" remote remove origin; \
		for fixture_root in \
			"$$task_root/repository/reference/kvquant/fixtures" \
			"$$task_root/repository/reference/kvquant_phase11pr/fixtures"; do \
			test -d "$$fixture_root"; \
			find "$$fixture_root" -type d -exec chmod 0555 {} +; \
			find "$$fixture_root" -type f -exec chmod 0444 {} +; \
		done; \
		test -d "$(PHASE11_KVQUANT_CALIBRATION)"; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env"; \
		git clone --quiet --no-local --no-checkout "$(PHASE11DQ23_KVQUANT_SOURCE_ROOT)" "$$task_root/kvquant-source"; \
		git -C "$$task_root/kvquant-source" checkout --quiet --detach "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		git -C "$$task_root/kvquant-source" remote remove origin; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD)" = "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD^{tree})" = "$(PHASE11DQ23_KVQUANT_TREE)"; \
		test -z "$$(git -C "$$task_root/kvquant-source" status --porcelain=v1 --untracked-files=all)"; \
		(cd "$$task_root/repository" && \
			PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$$task_root/repository/src:$$task_root/repository" \
				/usr/bin/python3 -m scripts.validate_kvquant_q23_long_context_patch \
					--source-root "$$task_root/kvquant-source" \
					>"$$task_root/source-validation.json"); \
		test "$$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["aggregate_patch_sha256"])' "$$task_root/source-validation.json")" = "$(PHASE11DQ23_KVQUANT_PATCH_SHA256)"; \
		mkdir "$$task_root/kvquant-build"; \
		for file in setup_cuda.py quant_cuda.cpp quant_cuda_kernel.cu measurement_cuda_kernel.cu; do \
			source="$$task_root/kvquant-source/deployment/kvquant/$$file"; \
			test -f "$$source" && test ! -L "$$source"; \
			cp -- "$$source" "$$task_root/kvquant-build/$$file"; \
		done; \
		mkdir -p "$(CURDIR)/artifacts/phase11"; \
		artifact_root="$$(mktemp -d "$(CURDIR)/artifacts/phase11/phase11dq23-launch.XXXXXX")"; \
		test -z "$$(find "$$artifact_root" -mindepth 1 -maxdepth 1 -print -quit)"; \
		cid="$$(docker create --read-only --network=none \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-source,dst=/opt/kvquant-source,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-build,dst=/opt/kvquant-build" \
			--mount "type=bind,src=$(PHASE11_KVQUANT_CALIBRATION),dst=/opt/kvquant-calibration,readonly" \
			--mount "type=bind,src=$$artifact_root,dst=/opt/phase11dq23-evidence" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TORCH_CUDA_ARCH_LIST=12.0+PTX \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$(PHASE11DQ23_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /usr/bin/bash "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" \
			--noprofile --norc -eu -o pipefail -c \
			'cd /opt/kvquant-build && /opt/kvbench/.venv/bin/python setup_cuda.py build_ext --inplace && extension="$$(find /opt/kvquant-build -maxdepth 1 -type f -name "quant_cuda.*.so" -print -quit)" && test -n "$$extension" && /usr/bin/strip --strip-unneeded "$$extension" && test "$$(sha256sum "$$extension" | cut -d " " -f 1)" = "$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)" && cd /home/rockrock/cmu_paper && /opt/kvbench/.venv/bin/python scripts/phase11dq23_kvquant_validation.py --repository-root /home/rockrock/cmu_paper --source-root /opt/kvquant-source --extension "$$extension" --fixture-root /home/rockrock/cmu_paper/reference/kvquant_phase11pr/fixtures --calibration-root /opt/kvquant-calibration --output-root /opt/phase11dq23-evidence' \
		)"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; \
		cid=""; \
		test -f "$$artifact_root/COMPLETE"; \
		(cd "$$artifact_root" && sha256sum --check --strict checksums.sha256 >/dev/null); \
		printf 'PHASE11DQ23_ARTIFACT=%s\n' "$$artifact_root"; \
		printf 'PHASE11DQ23_EVIDENCE_ROOT=%s\n' "$$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checksum_ledger_sha256"])' "$$artifact_root/COMPLETE")"

calibrate-kvquant: package-lock-check
	@$(MAKE) KVQUANT_GQA_SOURCE_ROOT="$(KVQUANT_CALIBRATION_SOURCE_ROOT)" validate-kvquant-gqa-patch
	@$(PHASE9_CALIBRATION) run --source-root "$(KVQUANT_CALIBRATION_SOURCE_ROOT)" --dataset-parquet "$(KVQUANT_CALIBRATION_DATASET_PARQUET)" --model-cache "$(KVQUANT_CALIBRATION_MODEL_CACHE)"

validate-calibration-kvquant:
	@$(PHASE9_CALIBRATION) validate $(if $(strip $(KVQUANT_CALIBRATION_ARTIFACT)),--artifact "$(KVQUANT_CALIBRATION_ARTIFACT)")

reference-kvquant: package-lock-check
	@$(MAKE) KVQUANT_GQA_SOURCE_ROOT="$(KVQUANT_REFERENCE_SOURCE_ROOT)" validate-kvquant-gqa-patch
	@$(MAKE) KVQUANT_CALIBRATION_ARTIFACT="$(KVQUANT_REFERENCE_CALIBRATION)" validate-calibration-kvquant
	@command -v docker >/dev/null
	@test "$$(docker image inspect "$(KVQUANT_REFERENCE_BASE_IMAGE)" --format '{{.Id}}')" = "$(KVQUANT_REFERENCE_BASE_IMAGE)"
	@test -f "$(KVQUANT_REFERENCE_TOKENIZERS_WHEEL)" && test ! -L "$(KVQUANT_REFERENCE_TOKENIZERS_WHEEL)"
	@test "$$(sha256sum "$(KVQUANT_REFERENCE_TOKENIZERS_WHEEL)" | cut -d ' ' -f 1)" = "$(KVQUANT_REFERENCE_TOKENIZERS_SHA256)"
	@if ! docker image inspect "$(KVQUANT_REFERENCE_IMAGE)" >/dev/null 2>&1; then \
		task_root="$$(mktemp -d /tmp/kvbench-reference-kvquant-build.XXXXXX)"; \
		trap 'rm -rf -- "$$task_root"' EXIT; \
		cp docker/reference-kvquant.Dockerfile "$$task_root/Dockerfile"; \
		cp "$(KVQUANT_REFERENCE_TOKENIZERS_WHEEL)" "$$task_root/"; \
		test "$$(find "$$task_root" -maxdepth 1 -type f | wc -l)" = 2; \
		test ! -e "$$task_root/.env"; \
		DOCKER_BUILDKIT=0 docker build --pull=false --network none \
			--build-arg BASE_IMAGE="$(KVQUANT_REFERENCE_BASE_IMAGE)" \
			--build-arg TOKENIZERS_WHEEL_SHA256="$(KVQUANT_REFERENCE_TOKENIZERS_SHA256)" \
			--tag "$(KVQUANT_REFERENCE_IMAGE_TAG)" \
			--file "$$task_root/Dockerfile" "$$task_root"; \
	fi
	@test "$$(docker image inspect "$(KVQUANT_REFERENCE_IMAGE)" --format '{{.Id}}')" = "$(KVQUANT_REFERENCE_IMAGE)"
	@build_root="$(KVQUANT_REFERENCE_BUILD_ROOT)"; \
		source_root="$(KVQUANT_REFERENCE_SOURCE_ROOT)"; \
		mkdir -p "$$build_root"; \
		for relative in quant_cuda.cpp quant_cuda_kernel.cu setup_cuda.py; do \
			source_file="$$source_root/deployment/kvquant/$$relative"; \
			build_file="$$build_root/$$relative"; \
			if test -e "$$build_file"; then \
				test -f "$$build_file" && test ! -L "$$build_file" && cmp -s "$$source_file" "$$build_file"; \
			else \
				cp -- "$$source_file" "$$build_file"; \
			fi; \
		done; \
		if test ! -e "$(KVQUANT_REFERENCE_EXTENSION)"; then \
			docker run --rm --network none --gpus all \
				--tmpfs /tmp:rw,exec,nosuid,size=2g \
				--mount type=bind,src="$$build_root",dst=/build \
				--workdir /build \
				-e PYTHONDONTWRITEBYTECODE=1 \
				-e PYTHONNOUSERSITE=1 \
				-e TORCH_CUDA_ARCH_LIST=12.0 \
				"$(KVQUANT_REFERENCE_IMAGE)" \
				/opt/kvbench/.venv/bin/python setup_cuda.py build_ext --inplace; \
		fi; \
		test -f "$(KVQUANT_REFERENCE_EXTENSION)" && test ! -L "$(KVQUANT_REFERENCE_EXTENSION)"; \
		test "$$(sha256sum "$(KVQUANT_REFERENCE_EXTENSION)" | cut -d ' ' -f 1)" = "$(KVQUANT_REFERENCE_EXTENSION_SHA256)"; \
		/usr/local/cuda/bin/cuobjdump --list-elf "$(KVQUANT_REFERENCE_EXTENSION)" | grep -F '.sm_120.cubin' >/dev/null; \
		/usr/local/cuda/bin/cuobjdump --dump-ptx "$(KVQUANT_REFERENCE_EXTENSION)" | grep -F '.version 9.0' >/dev/null; \
		/usr/local/cuda/bin/cuobjdump --dump-ptx "$(KVQUANT_REFERENCE_EXTENSION)" | grep -F '.target sm_120' >/dev/null
	@docker run --rm --network none --gpus all \
		--tmpfs /tmp:rw,exec,nosuid,size=2g \
		--mount type=bind,src="$(CURDIR)",dst=/repo,readonly \
		--mount type=bind,src="$(KVQUANT_REFERENCE_SOURCE_ROOT)",dst=/source,readonly \
		--mount type=bind,src="$(KVQUANT_REFERENCE_BUILD_ROOT)",dst=/build,readonly \
		--mount type=bind,src="$(KVQUANT_REFERENCE_CALIBRATION)",dst=/calibration,readonly \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e PYTHONNOUSERSITE=1 \
		-e PYTHONPATH=/source/deployment/transformers/src:/source:/build:/repo:/opt/kvbench-reference/deps:/opt/kvbench/.phase3/site-packages \
		-e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		"$(KVQUANT_REFERENCE_IMAGE)" \
		/opt/kvbench/.venv/bin/python /repo/tests/cuda/phase10_kvquant_sanitizer_probe.py \
			--source-root /source \
			--calibration-root /calibration \
			--patch-manifest /repo/third_party/patches/kvquant/manifest.json
	@docker run --rm --network none --gpus all \
		--tmpfs /tmp:rw,exec,nosuid,size=2g \
		--mount type=bind,src="$(KVQUANT_REFERENCE_SOURCE_ROOT)",dst=/source,readonly \
		--mount type=bind,src="$(KVQUANT_REFERENCE_BUILD_ROOT)",dst=/build,readonly \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e PYTHONNOUSERSITE=1 \
		-e PYTHONPATH=/source:/build:/opt/kvbench-reference/deps:/opt/kvbench/.phase3/site-packages \
		-e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		-e CUDA_FORCE_PTX_JIT=1 \
		-e CUDA_CACHE_DISABLE=1 \
		-e KVQUANT_RUN_PTX_TESTS=1 \
		-e KVQUANT_CUDA_EXTENSION_DIR=/build \
		"$(KVQUANT_REFERENCE_IMAGE)" \
		/opt/kvbench/.venv/bin/python /source/tests_phase9p/test_ptx_jit.py -v
	@docker run --rm --network none --gpus all \
		--tmpfs /tmp:rw,exec,nosuid,size=2g \
		--mount type=bind,src="$(CURDIR)",dst=/repo,readonly \
		--mount type=bind,src="$(KVQUANT_REFERENCE_SOURCE_ROOT)",dst=/source,readonly \
		--mount type=bind,src="$(KVQUANT_REFERENCE_BUILD_ROOT)",dst=/build,readonly \
		--mount type=bind,src="$(KVQUANT_REFERENCE_CALIBRATION)",dst=/calibration,readonly \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e PYTHONNOUSERSITE=1 \
		-e PYTHONPATH=/source/deployment/transformers/src:/source:/build:/repo:/opt/kvbench-reference/deps:/opt/kvbench/.phase3/site-packages \
		-e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		--entrypoint /usr/local/cuda-13.0/bin/compute-sanitizer \
		"$(KVQUANT_REFERENCE_IMAGE)" \
		--tool memcheck --leak-check full --error-exitcode=86 \
		--target-processes application-only \
		/opt/kvbench/.venv/bin/python /repo/tests/cuda/phase10_kvquant_sanitizer_probe.py \
			--source-root /source \
			--calibration-root /calibration \
			--patch-manifest /repo/third_party/patches/kvquant/manifest.json
	@if ! test -e reference/kvquant/fixtures; then \
		docker run --rm --network none --gpus all \
			--tmpfs /tmp:rw,exec,nosuid,size=2g \
			--mount type=bind,src="$(CURDIR)",dst=/repo,readonly \
			--mount type=bind,src="$(CURDIR)/reference/kvquant",dst=/output \
			--mount type=bind,src="$(KVQUANT_REFERENCE_SOURCE_ROOT)",dst=/source,readonly \
			--mount type=bind,src="$(KVQUANT_REFERENCE_BUILD_ROOT)",dst=/build,readonly \
			--mount type=bind,src="$(KVQUANT_REFERENCE_CALIBRATION)",dst=/calibration,readonly \
			-e PYTHONDONTWRITEBYTECODE=1 \
			-e PYTHONNOUSERSITE=1 \
			-e PYTHONPATH=/source/deployment/transformers/src:/source:/build:/repo:/opt/kvbench-reference/deps:/opt/kvbench/.phase3/site-packages \
			-e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			"$(KVQUANT_REFERENCE_IMAGE)" \
			/opt/kvbench/.venv/bin/python /repo/reference/kvquant/generate_fixtures.py fixtures \
				--source-root /source \
				--calibration-root /calibration \
				--patch-manifest /repo/third_party/patches/kvquant/manifest.json \
				--extension /build/quant_cuda.cpython-312-x86_64-linux-gnu.so \
				--reference-root /output; \
	fi
	@$(MAKE) validate-reference-kvquant
	@determinism_root="$$(mktemp -d /tmp/kvbench-phase10-determinism.XXXXXX)"; \
	trap 'chmod -R u+w "$$determinism_root" 2>/dev/null || true; rm -rf -- "$$determinism_root"' EXIT; \
		docker run --rm --network none --gpus all \
			--tmpfs /tmp:rw,exec,nosuid,size=2g \
			--mount type=bind,src="$(CURDIR)",dst=/repo,readonly \
			--mount type=bind,src="$$determinism_root",dst=/output \
			--mount type=bind,src="$(KVQUANT_REFERENCE_SOURCE_ROOT)",dst=/source,readonly \
			--mount type=bind,src="$(KVQUANT_REFERENCE_BUILD_ROOT)",dst=/build,readonly \
			--mount type=bind,src="$(KVQUANT_REFERENCE_CALIBRATION)",dst=/calibration,readonly \
			-e PYTHONDONTWRITEBYTECODE=1 \
			-e PYTHONNOUSERSITE=1 \
			-e PYTHONPATH=/source/deployment/transformers/src:/source:/build:/repo:/opt/kvbench-reference/deps:/opt/kvbench/.phase3/site-packages \
			-e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			"$(KVQUANT_REFERENCE_IMAGE)" \
			/opt/kvbench/.venv/bin/python /repo/reference/kvquant/generate_fixtures.py fixtures \
				--source-root /source \
				--calibration-root /calibration \
				--patch-manifest /repo/third_party/patches/kvquant/manifest.json \
				--extension /build/quant_cuda.cpython-312-x86_64-linux-gnu.so \
				--reference-root /output; \
		diff -qr reference/kvquant/fixtures "$$determinism_root/fixtures"; \
		for manifest in source_manifest.json environment.json calibration_manifest.json build_manifest.json; do \
			cmp -s "reference/kvquant/$$manifest" "$$determinism_root/$$manifest"; \
		done

validate-reference-kvquant:
	@docker run --rm --network none \
		--tmpfs /tmp:rw,exec,nosuid,size=1g \
		--mount type=bind,src="$(CURDIR)",dst=/repo,readonly \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e PYTHONNOUSERSITE=1 \
		-e PYTHONPATH=/repo:/opt/kvbench-reference/deps:/opt/kvbench/.phase3/site-packages \
		"$(KVQUANT_REFERENCE_IMAGE)" \
		/opt/kvbench/.venv/bin/python /repo/reference/kvquant/validate_fixtures.py \
			--fixtures /repo/reference/kvquant/fixtures
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -c 'from scripts.r2_artifact import validate_local_artifact; artifact = validate_local_artifact("reference/kvquant/fixtures"); print("{\"status\":\"PASS\",\"local_root_sha256\":\"%s\",\"object_count\":%d}" % (artifact.root_sha256, len(artifact.files)))'

validate-reference-kvquant-phase11pr:
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m reference.kvquant_phase11pr.validate_corrected_bundle --fixtures "$(KVQUANT_PHASE11PR_FIXTURES)" --old-fixtures "$(CURDIR)/reference/kvquant/fixtures"
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -c 'from scripts.r2_artifact import validate_local_artifact; artifact = validate_local_artifact("reference/kvquant_phase11pr/fixtures"); print("{\"status\":\"PASS\",\"local_root_sha256\":\"%s\",\"object_count\":%d}" % (artifact.root_sha256, len(artifact.files)))'

admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_SOURCE_ROOT := $(PHASE11DQ23_KVQUANT_SOURCE_ROOT)
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_COMMIT := $(PHASE11DQ23_KVQUANT_COMMIT)
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_TREE := $(PHASE11DQ23_KVQUANT_TREE)
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_PATCH_SHA256 := $(PHASE11DQ23_KVQUANT_PATCH_SHA256)
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_EXTENSION_SHA256 := $(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_VALIDATOR_PATH := scripts/validate_kvquant_q23_long_context_patch.py
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_VALIDATOR_MODULE := scripts.validate_kvquant_q23_long_context_patch
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE := decision0029
admit-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_LAUNCH_PREFIX := phase11rq23-launch
admit-kvquant admit-kvquant-q23: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)
admit-kvquant admit-kvquant-q23: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase11_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase11-kvquant-admission.XXXXXX)"; \
		cid=""; artifact_root=""; preserve=1; q23_mount=(); \
		cleanup() { \
			if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$artifact_root" ]]; then \
				printf '{"status":"PHASE11_LAUNCH_ARTIFACT_ROOT_PRESERVED","path":"%s"}\n' "$$artifact_root" >&2; \
			fi; \
			if (( preserve == 0 )); then \
				chmod -R u+w "$$task_root" 2>/dev/null || true; \
				rm -rf -- "$$task_root"; \
			else \
				printf '{"status":"FAILED_PHASE11_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; \
			fi; \
		}; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test "$$repository_root" = "$(CURDIR)"; \
		authority_profile="$(PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE)"; \
		if test "$$authority_profile" = decision0029; then \
			q23_evidence="$$(realpath -e "$(PHASE11RQ23_Q23_EVIDENCE)")"; \
			test "$$q23_evidence" = "$(PHASE11RQ23_Q23_EVIDENCE)"; \
			test -d "$$q23_evidence" && test ! -L "$(PHASE11RQ23_Q23_EVIDENCE)"; \
			test ! -e "$$q23_evidence/.env" && test ! -L "$$q23_evidence/.env"; \
			q23_mount=( \
				--mount "type=bind,src=$$q23_evidence,dst=/opt/phase11dq23-evidence,readonly" \
				--env KVBENCH_PHASE11DQ23_EVIDENCE_ROOT=/opt/phase11dq23-evidence \
			); \
		else \
			test "$$authority_profile" = decision0027; \
		fi; \
		image_id="$(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)"; \
		test "$$(docker image inspect "$$image_id" --format '{{.Id}}')" = "$$image_id"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/repository" remote remove origin; \
		test "$$(git -C "$$task_root/repository" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env"; \
		for fixture_root in \
			"$$task_root/repository/reference/kvquant/fixtures" \
			"$$task_root/repository/reference/kvquant_phase11pr/fixtures"; do \
			test -d "$$fixture_root" && test ! -L "$$fixture_root"; \
			chmod -R a-w -- "$$fixture_root"; \
			test -z "$$(find "$$fixture_root" -perm /222 -print -quit)"; \
		done; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		for file in \
			scripts/phase11_kvquant_admission.py \
			$(PHASE11_KVQUANT_ACTIVE_VALIDATOR_PATH) \
			src/kvbench/adapters/kvquant.py \
			src/kvbench/runtime/bf16_endpoint.py \
			src/kvbench/runtime/kvquant_cache.py \
			src/kvbench/runtime/kvquant_session.py \
			src/kvbench/schema/phase11.py \
			tests/cuda/phase11_kvquant_sanitizer_probe.py \
			tests/cuda/test_phase11_kvquant_cuda.py \
			tests/graph/test_phase11_kvquant_graph.py; do \
			target="$$task_root/repository/$$file"; \
			test -f "$$target" && test ! -L "$$target"; \
			git -C "$$task_root/repository" cat-file -e "$$head:$$file"; \
			committed_sha256="$$(git -C "$$task_root/repository" cat-file blob "$$head:$$file" | sha256sum | cut -d " " -f 1)"; \
			test "$$(sha256sum "$$target" | cut -d " " -f 1)" = "$$committed_sha256"; \
		done; \
		mkdir "$$task_root/repository/.venv" "$$task_root/repository/.phase3"; \
		ln -s /opt/kvbench/.venv/bin "$$task_root/repository/.venv/bin"; \
		ln -s /opt/kvbench/.venv/lib "$$task_root/repository/.venv/lib"; \
		ln -s /opt/kvbench/.venv/pyvenv.cfg "$$task_root/repository/.venv/pyvenv.cfg"; \
		ln -s /opt/kvbench/.phase3/site-packages "$$task_root/repository/.phase3/site-packages"; \
		git clone --quiet --no-local --no-checkout "$(PHASE11_KVQUANT_ACTIVE_SOURCE_ROOT)" "$$task_root/kvquant-source"; \
		git -C "$$task_root/kvquant-source" checkout --quiet --detach "$(PHASE11_KVQUANT_ACTIVE_COMMIT)"; \
		git -C "$$task_root/kvquant-source" remote remove origin; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD)" = "$(PHASE11_KVQUANT_ACTIVE_COMMIT)"; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD^{tree})" = "$(PHASE11_KVQUANT_ACTIVE_TREE)"; \
		test -z "$$(git -C "$$task_root/kvquant-source" status --porcelain=v1 --untracked-files=all)"; \
		/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$(CURDIR):$(CURDIR)/src" /usr/bin/python3 -m $(PHASE11_KVQUANT_ACTIVE_VALIDATOR_MODULE) --source-root "$$task_root/kvquant-source" > "$$task_root/source-validation.json"; \
		test "$$(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC /usr/bin/python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["aggregate_patch_sha256"])' "$$task_root/source-validation.json")" = "$(PHASE11_KVQUANT_ACTIVE_PATCH_SHA256)"; \
		mkdir "$$task_root/kvquant-build"; \
		for file in setup_cuda.py quant_cuda.cpp quant_cuda_kernel.cu measurement_cuda_kernel.cu; do \
			relative="deployment/kvquant/$$file"; \
			source="$$task_root/kvquant-source/$$relative"; \
			test -f "$$source" && test ! -L "$$source"; \
			git -C "$$task_root/kvquant-source" cat-file -e "$(PHASE11_KVQUANT_ACTIVE_COMMIT):$$relative"; \
			committed_sha256="$$(git -C "$$task_root/kvquant-source" cat-file blob "$(PHASE11_KVQUANT_ACTIVE_COMMIT):$$relative" | sha256sum | cut -d " " -f 1)"; \
			test "$$(sha256sum "$$source" | cut -d " " -f 1)" = "$$committed_sha256"; \
			cp -- "$$source" "$$task_root/kvquant-build/$$file"; \
			test "$$(sha256sum "$$task_root/kvquant-build/$$file" | cut -d " " -f 1)" = "$$committed_sha256"; \
		done; \
		calibration_root="$(PHASE11_KVQUANT_CALIBRATION)"; \
		test "$$calibration_root" = "$$repository_root/calibration/kvquant/kvqcal-cdb724c806d64d095c040d2673a987a3"; \
		test -d "$$calibration_root" && test ! -L "$$calibration_root"; \
		test "$$(realpath -e "$$calibration_root")" = "$$calibration_root"; \
		calibration_digest="$$(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$(CURDIR):$(CURDIR)/src" /usr/bin/python3 -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1], environ={}).root_sha256)' "$$calibration_root")"; \
		test "$$calibration_digest" = "$(PHASE11_KVQUANT_CALIBRATION_ROOT_SHA256)"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		model_snapshot="$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		test -d "$$model_root" && test -d "$$model_snapshot"; \
		artifact_parent="$$repository_root/artifacts"; \
		if test -e "$$artifact_parent" || test -L "$$artifact_parent"; then \
			test -d "$$artifact_parent" && test ! -L "$$artifact_parent"; \
		else \
			mkdir -- "$$artifact_parent"; \
		fi; \
		artifact_store_root="$$artifact_parent/phase11"; \
		if test -e "$$artifact_store_root" || test -L "$$artifact_store_root"; then \
			test -d "$$artifact_store_root" && test ! -L "$$artifact_store_root"; \
		else \
			mkdir -- "$$artifact_store_root"; \
		fi; \
		test "$$(realpath -e "$$artifact_store_root")" = "$$repository_root/artifacts/phase11"; \
		artifact_root="$$(mktemp -d "$$artifact_store_root/$(PHASE11_KVQUANT_ACTIVE_LAUNCH_PREFIX).XXXXXX")"; \
		test -d "$$artifact_root" && test ! -L "$$artifact_root"; \
		test "$$(realpath -e "$$artifact_root/..")" = "$$artifact_store_root"; \
		test ! -e "$$artifact_root/.env" && test ! -L "$$artifact_root/.env"; \
		test -z "$$(find "$$artifact_root" -mindepth 1 -maxdepth 1 -print -quit)"; \
		mkdir -p "$$task_root/repository/artifacts/phase11"; \
		cid="$$(docker create --read-only --network=none \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$artifact_root,dst=/home/rockrock/cmu_paper/artifacts/phase11" \
			--mount "type=bind,src=$$task_root/kvquant-source,dst=/opt/kvquant-source,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-build,dst=/opt/kvquant-build" \
			"$${q23_mount[@]}" \
			--mount "type=bind,src=$$calibration_root,dst=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3,readonly" \
			--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env HF_HUB_OFFLINE=1 \
			--env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 \
			--env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TORCH_CUDA_ARCH_LIST=12.0+PTX \
			--env KVBENCH_KVQUANT_SOURCE_ROOT=/opt/kvquant-source \
			--env KVBENCH_KVQUANT_CALIBRATION_ROOT=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3 \
			--env KVBENCH_KVQUANT_EXTENSION_SHA256="$(PHASE11_KVQUANT_ACTIVE_EXTENSION_SHA256)" \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /usr/bin/bash "$$image_id" \
			--noprofile --norc -eu -o pipefail -c \
			'cd /opt/kvquant-build && /opt/kvbench/.venv/bin/python setup_cuda.py build_ext --inplace && fresh_extension="$$(find /opt/kvquant-build -maxdepth 1 -type f -name "quant_cuda.*.so" -print -quit)" && test -n "$$fresh_extension" && /usr/bin/strip --strip-unneeded "$$fresh_extension" && test "$$(sha256sum "$$fresh_extension" | cut -d " " -f 1)" = "$(PHASE11_KVQUANT_ACTIVE_EXTENSION_SHA256)" && /usr/local/cuda-13.0/bin/cuobjdump --list-elf "$$fresh_extension" | grep -F ".sm_120.cubin" >/dev/null && /usr/local/cuda-13.0/bin/cuobjdump --dump-ptx "$$fresh_extension" | grep -F ".target sm_120" >/dev/null && export KVBENCH_KVQUANT_EXTENSION="$$fresh_extension" KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION="$$fresh_extension" && /opt/kvbench/.venv/bin/python -m $(PHASE11_KVQUANT_ACTIVE_VALIDATOR_MODULE) --source-root /opt/kvquant-source && cd /home/rockrock/cmu_paper && /opt/kvbench/.venv/bin/python -m scripts.phase11_kvquant_admission --authority-profile "$(PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE)"')"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		preserve=0

validate-admission-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE := decision0029
validate-admission-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_INNER_ARTIFACT := $(PHASE11RQ23_KVQUANT_INNER_ARTIFACT)
validate-admission-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_OUTER_ARTIFACT := $(PHASE11RQ23_KVQUANT_OUTER_ARTIFACT)
validate-admission-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_REPORT := $(PHASE11RQ23_KVQUANT_METHOD_ADMISSION_REPORT)
validate-admission-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_CHECKSUM := $(PHASE11RQ23_KVQUANT_METHOD_ADMISSION_CHECKSUM)
validate-admission-kvquant-q23: override PHASE11_KVQUANT_ACTIVE_PUBLICATION_RECEIPT := $(PHASE11RQ23_KVQUANT_PUBLICATION_RECEIPT)
validate-admission-kvquant validate-admission-kvquant-q23: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)
validate-admission-kvquant validate-admission-kvquant-q23: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase11_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@test -n "$(PHASE11_KVQUANT_ACTIVE_INNER_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"phase11_inner_artifact_required"}' >&2; exit 2; }
	@test -n "$(PHASE11_KVQUANT_ACTIVE_OUTER_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"phase11_outer_artifact_required"}' >&2; exit 2; }
	@test -f "$(PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_REPORT)" || { echo '{"status":"BLOCKED","reason":"phase11_method_admission_report_required"}' >&2; exit 2; }
	@test -f "$(PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_CHECKSUM)" || { echo '{"status":"BLOCKED","reason":"phase11_method_admission_checksum_required"}' >&2; exit 2; }
	@test -f "$(PHASE11_KVQUANT_ACTIVE_PUBLICATION_RECEIPT)" || { echo '{"status":"BLOCKED","reason":"phase11_publication_receipt_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase11-final-validation.XXXXXX)"; \
		cid=""; \
		cleanup() { \
			if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; \
			chmod -R u+w "$$task_root" 2>/dev/null || true; \
			rm -rf -- "$$task_root"; \
		}; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test "$$repository_root" = "$(CURDIR)"; \
		image_id="$(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)"; \
		test "$$(docker image inspect "$$image_id" --format '{{.Id}}')" = "$$image_id"; \
		inner_source="$$(realpath -e "$(PHASE11_KVQUANT_ACTIVE_INNER_ARTIFACT)")"; \
		outer_source="$$(realpath -e "$(PHASE11_KVQUANT_ACTIVE_OUTER_ARTIFACT)")"; \
		test -d "$$inner_source" && test ! -L "$(PHASE11_KVQUANT_ACTIVE_INNER_ARTIFACT)"; \
		test -d "$$outer_source" && test ! -L "$(PHASE11_KVQUANT_ACTIVE_OUTER_ARTIFACT)"; \
		case "$$inner_source" in "$$repository_root"/artifacts/phase11/*) ;; *) echo '{"status":"BLOCKED","reason":"phase11_inner_artifact_must_be_repository_relative"}' >&2; exit 2;; esac; \
		case "$$outer_source" in "$$repository_root"/artifacts/phase11_r2_outer/*) ;; *) echo '{"status":"BLOCKED","reason":"phase11_outer_artifact_must_be_repository_relative"}' >&2; exit 2;; esac; \
		inner_relative="$${inner_source#$$repository_root/}"; \
		outer_relative="$${outer_source#$$repository_root/}"; \
		inner_name="$$(basename "$$inner_source")"; \
		outer_name="$$(basename "$$outer_source")"; \
		[[ "$$inner_name" =~ ^[a-z0-9][a-z0-9._-]{0,127}$$ ]]; \
		[[ "$$outer_name" =~ ^[a-z0-9][a-z0-9._-]{0,127}$$ ]]; \
		/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$(CURDIR):$(CURDIR)/src" /usr/bin/python3 -c 'import sys; from scripts.r2_artifact import validate_local_artifact; artifact=validate_local_artifact(sys.argv[1],environ={}); print(artifact.root_sha256)' "$$inner_source" > "$$task_root/inner-root.sha256"; \
		/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$(CURDIR):$(CURDIR)/src" /usr/bin/python3 -c 'import sys; from scripts.r2_artifact import validate_local_artifact; artifact=validate_local_artifact(sys.argv[1],environ={}); print(artifact.root_sha256)' "$$outer_source" > "$$task_root/outer-root.sha256"; \
		grep -Eq '^[0-9a-f]{64}$$' "$$task_root/inner-root.sha256"; \
		grep -Eq '^[0-9a-f]{64}$$' "$$task_root/outer-root.sha256"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/repository" remote remove origin; \
		test "$$(git -C "$$task_root/repository" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env"; \
		test -f "$$task_root/repository/$(PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_REPORT)"; \
		test -f "$$task_root/repository/$(PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_CHECKSUM)"; \
		test -f "$$task_root/repository/$(PHASE11_KVQUANT_ACTIVE_PUBLICATION_RECEIPT)"; \
		(cd "$$task_root/repository/$$(dirname "$(PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_CHECKSUM)")" && /usr/bin/sha256sum -c "$$(basename "$(PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_CHECKSUM)")"); \
		mkdir -p "$$task_root/repository/$$(dirname "$$inner_relative")" "$$task_root/repository/$$(dirname "$$outer_relative")"; \
		/usr/bin/cp --archive --reflink=auto "$$inner_source" "$$task_root/repository/$$(dirname "$$inner_relative")/"; \
		/usr/bin/cp --archive --reflink=auto "$$outer_source" "$$task_root/repository/$$(dirname "$$outer_relative")/"; \
		test "$$(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$$task_root/repository:$$task_root/repository/src" /usr/bin/python3 -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1],environ={}).root_sha256)' "$$task_root/repository/$$inner_relative")" = "$$(cat "$$task_root/inner-root.sha256")"; \
		test "$$(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$$task_root/repository:$$task_root/repository/src" /usr/bin/python3 -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1],environ={}).root_sha256)' "$$task_root/repository/$$outer_relative")" = "$$(cat "$$task_root/outer-root.sha256")"; \
		cid="$$(docker create --read-only --network=none \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=1g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper:/opt/kvbench/.phase3/site-packages \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /opt/kvbench/.venv/bin/python "$$image_id" \
			-m scripts.phase11_kvquant_admission \
			--authority-profile "$(PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE)" \
			--validate-only \
			--artifact "/home/rockrock/cmu_paper/$$inner_relative" \
			--outer-artifact "/home/rockrock/cmu_paper/$$outer_relative" \
			--method-admission-report "/home/rockrock/cmu_paper/$(PHASE11_KVQUANT_ACTIVE_METHOD_ADMISSION_REPORT)" \
			--publication-receipt "/home/rockrock/cmu_paper/$(PHASE11_KVQUANT_ACTIVE_PUBLICATION_RECEIPT)")"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""

unified-admission: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE12_AUTHORIZED_IMAGE_CONFIG_DIGEST)
unified-admission: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE12_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase12_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"clean_committed_phase12_tree_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase12-unified.XXXXXX)"; \
		cid=""; reference_cid=""; stage=""; campaign_id=""; final_root=""; head=""; preserve=1; \
		cleanup() { \
			status=$$?; \
			if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$reference_cid" ]]; then docker rm -f "$$reference_cid" >/dev/null 2>&1 || true; fi; \
			if (( status != 0 )) && [[ -n "$$stage" && -d "$$stage" && -n "$$campaign_id" && -n "$$head" ]]; then \
				if failed_result="$$($(PHASE12_HOST_PYTHON) -m scripts.phase12_unified_admission --finalize-failed-campaign --stage "$$stage" --campaign-id "$$campaign_id" --git-sha "$$head" --failure-code "$$status" 2>&1)"; then \
					printf '%s\n' "$$failed_result" >&2; \
					stage=""; \
				else \
					failed_detail="$$($(PHASE12_HOST_PYTHON) -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$$failed_result")"; \
					printf '{"status":"PHASE12_TERMINAL_FAILURE_FINALIZATION_FAILED","detail":%s}\n' "$$failed_detail" >&2; \
				fi; \
			fi; \
			if [[ -n "$$stage" && -d "$$stage" ]]; then \
				printf '{"status":"PHASE12_STAGING_EVIDENCE_PRESERVED","path":"%s"}\n' "$$stage" >&2; \
			fi; \
			if (( preserve == 0 )); then \
				chmod -R u+w "$$task_root" 2>/dev/null || true; \
				rm -rf -- "$$task_root"; \
			else \
				printf '{"status":"FAILED_PHASE12_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; \
			fi; \
			trap - EXIT; \
			exit $$status; \
		}; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test "$$repository_root" = "$(CURDIR)"; \
		git cat-file -e "$$head:docs/plans/phase12-unified-admission.md"; \
		image_id="$(PHASE12_AUTHORIZED_IMAGE_CONFIG_DIGEST)"; \
		test "$$(docker image inspect "$$image_id" --format '{{.Id}}')" = "$$image_id"; \
		campaign_id="$$($(PHASE12_HOST_PYTHON) -m scripts.phase12_unified_admission --new-campaign-id --git-sha "$$head")"; \
		[[ "$$campaign_id" =~ ^phase12-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}$$ ]]; \
		stage="$$($(PHASE12_HOST_PYTHON) -m scripts.phase12_unified_admission --reserve-campaign --campaign-id "$$campaign_id" --git-sha "$$head")"; \
		test -d "$$stage" && test ! -L "$$stage"; \
		test "$$(realpath -e "$$stage")" = "$$stage"; \
		case "$$stage" in "$$repository_root"/artifacts/phase12/.kvbench-staging/"$$campaign_id".*.staging) ;; *) echo '{"status":"BLOCKED","reason":"unsafe_phase12_stage"}' >&2; exit 2;; esac; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/repository" remote remove origin; \
		test "$$(git -C "$$task_root/repository" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env" && test ! -L "$$task_root/repository/.env"; \
		mkdir "$$task_root/repository/.venv" "$$task_root/repository/.phase3"; \
		ln -s /opt/kvbench/.venv/bin "$$task_root/repository/.venv/bin"; \
		ln -s /opt/kvbench/.venv/lib "$$task_root/repository/.venv/lib"; \
		ln -s /opt/kvbench/.venv/pyvenv.cfg "$$task_root/repository/.venv/pyvenv.cfg"; \
		ln -s /opt/kvbench/.phase3/site-packages "$$task_root/repository/.phase3/site-packages"; \
		stage_relative="$${stage#$$repository_root/}"; \
		test "$$stage_relative" != "$$stage"; \
		mkdir -p "$$task_root/repository/$$(dirname "$$stage_relative")"; \
		mkdir "$$task_root/repository/$$stage_relative"; \
		for immutable_relative in docs/evidence/e00 reference/kvquant/fixtures reference/kvquant_phase11pr/fixtures; do \
			immutable_root="$$task_root/repository/$$immutable_relative"; \
			test -d "$$immutable_root" && test ! -L "$$immutable_root"; \
			chmod -R a-w -- "$$immutable_root"; \
			test -z "$$(find "$$immutable_root" -perm /222 -print -quit)"; \
		done; \
		phase12_artifact_root="$$repository_root/artifacts/phase12"; \
		test -d "$$phase12_artifact_root" && test ! -L "$$phase12_artifact_root"; \
		test "$$(realpath -e "$$phase12_artifact_root")" = "$$phase12_artifact_root"; \
		reference_image="$(KIVI_REFERENCE_IMAGE)@$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$reference_image" --format '{{.Id}}')" = "$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$reference_image" --format '{{index .Config.Labels "org.kvbench.reference.parent.config_digest"}}')" = "$$image_id"; \
		test "$$($(PHASE2_PYTHON) -c 'import json; print(json.load(open("reference/kivi/build_manifest.json", encoding="utf-8"))["image"]["config_digest"])')" = "$(PHASE8_KIVI_REFERENCE_CONFIG_DIGEST)"; \
		mkdir "$$task_root/kivi-source" "$$task_root/kivi-extension"; \
		reference_cid="$$(docker create --network=none "$$reference_image")"; \
		[[ "$$reference_cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker cp "$$reference_cid:/opt/kivi-source/." "$$task_root/kivi-source"; \
		docker cp "$$reference_cid:/opt/kivi-source/quant/kivi_gemv.cpython-312-x86_64-linux-gnu.so" "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so"; \
		docker rm -f "$$reference_cid" >/dev/null; reference_cid=""; \
		test -f "$$task_root/kivi-source/models/kivi_gqa.py" && test ! -L "$$task_root/kivi-source/models/kivi_gqa.py"; \
		cp --preserve=mode,timestamps "$$task_root/kivi-source/models/kivi_gqa.py" "$$task_root/kivi-gqa.authority.py"; \
		git -C "$$task_root/kivi-source" clean -fdx; \
		cp --preserve=mode,timestamps "$$task_root/kivi-gqa.authority.py" "$$task_root/kivi-source/models/kivi_gqa.py"; \
		test "$$(sha256sum "$$task_root/kivi-source/quant/new_pack.py" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_NEW_PACK_SHA256)"; \
		test "$$(sha256sum "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_EXTENSION_SHA256)"; \
		$(PHASE12_HOST_PYTHON) scripts/validate_kivi_b019_patch.py --device cpu --source-root "$$task_root/kivi-source" > "$$task_root/kivi-source-validation.json"; \
		git clone --quiet --no-local --no-checkout "$(PHASE11DQ23_KVQUANT_SOURCE_ROOT)" "$$task_root/kvquant-source"; \
		git -C "$$task_root/kvquant-source" checkout --quiet --detach "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		git -C "$$task_root/kvquant-source" remote remove origin; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD)" = "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD^{tree})" = "$(PHASE11DQ23_KVQUANT_TREE)"; \
		test -z "$$(git -C "$$task_root/kvquant-source" status --porcelain=v1 --untracked-files=all)"; \
		$(PHASE12_HOST_PYTHON) -m scripts.validate_kvquant_q23_long_context_patch --source-root "$$task_root/kvquant-source" > "$$task_root/kvquant-source-validation.json"; \
		test "$$($(PHASE2_PYTHON) -I -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["aggregate_patch_sha256"])' "$$task_root/kvquant-source-validation.json")" = "$(PHASE11DQ23_KVQUANT_PATCH_SHA256)"; \
		mkdir "$$task_root/kvquant-build"; \
		for file in setup_cuda.py quant_cuda.cpp quant_cuda_kernel.cu measurement_cuda_kernel.cu; do \
			relative="deployment/kvquant/$$file"; \
			source="$$task_root/kvquant-source/$$relative"; \
			test -f "$$source" && test ! -L "$$source"; \
			committed_sha256="$$(git -C "$$task_root/kvquant-source" cat-file blob "$(PHASE11DQ23_KVQUANT_COMMIT):$$relative" | sha256sum | cut -d " " -f 1)"; \
			test "$$(sha256sum "$$source" | cut -d " " -f 1)" = "$$committed_sha256"; \
			cp -- "$$source" "$$task_root/kvquant-build/$$file"; \
			test "$$(sha256sum "$$task_root/kvquant-build/$$file" | cut -d " " -f 1)" = "$$committed_sha256"; \
		done; \
		calibration_root="$(PHASE11_KVQUANT_CALIBRATION)"; \
		test -d "$$calibration_root" && test ! -L "$$calibration_root"; \
		test "$$(realpath -e "$$calibration_root")" = "$$calibration_root"; \
		calibration_digest="$$(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$(CURDIR):$(CURDIR)/src" /usr/bin/python3 -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1], environ={}).root_sha256)' "$$calibration_root")"; \
		test "$$calibration_digest" = "$(PHASE11_KVQUANT_CALIBRATION_ROOT_SHA256)"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		model_snapshot="$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		test -d "$$model_root" && test -d "$$model_snapshot"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$phase12_artifact_root,dst=/home/rockrock/cmu_paper/artifacts/phase12,readonly" \
			--mount "type=bind,src=$$stage,dst=/home/rockrock/cmu_paper/$$stage_relative" \
			--mount "type=bind,src=$$task_root/kivi-source,dst=/opt/kivi-source,readonly" \
			--mount "type=bind,src=$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so,dst=/opt/kvbench/.phase3/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-source,dst=/opt/kvquant-source,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-build,dst=/opt/kvquant-build" \
			--mount "type=bind,src=$$calibration_root,dst=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3,readonly" \
			--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONIOENCODING=utf-8 \
			--env PYTHONPATH=/opt/kivi-source:/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env LANG=C.UTF-8 \
			--env TZ=UTC \
			--env HF_HUB_OFFLINE=1 \
			--env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 \
			--env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TORCH_CUDA_ARCH_LIST=12.0+PTX \
			--env TRITON_CACHE_DIR=/root/.triton \
			--env KVBENCH_KIVI_SOURCE_ROOT=/opt/kivi-source \
			--env KVBENCH_KVQUANT_SOURCE_ROOT=/opt/kvquant-source \
			--env KVBENCH_KVQUANT_CALIBRATION_ROOT=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3 \
			--env KVBENCH_KVQUANT_EXTENSION_SHA256="$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)" \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /usr/bin/bash "$$image_id" \
			--noprofile --norc -eu -o pipefail -c \
			'cd /opt/kvquant-build && /opt/kvbench/.venv/bin/python setup_cuda.py build_ext --inplace && fresh_extension="$$(find /opt/kvquant-build -maxdepth 1 -type f -name "quant_cuda.*.so" -print -quit)" && test -n "$$fresh_extension" && /usr/bin/strip --strip-unneeded "$$fresh_extension" && test "$$(sha256sum "$$fresh_extension" | cut -d " " -f 1)" = "$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)" && /usr/local/cuda-13.0/bin/cuobjdump --list-elf "$$fresh_extension" | grep -F ".sm_120.cubin" >/dev/null && /usr/local/cuda-13.0/bin/cuobjdump --dump-ptx "$$fresh_extension" | grep -F ".target sm_120" >/dev/null && export KVBENCH_KVQUANT_EXTENSION="$$fresh_extension" KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION="$$fresh_extension" && cd /home/rockrock/cmu_paper && /usr/bin/mkdir -p /tmp/kivi-b019-objects && GIT_OBJECT_DIRECTORY=/tmp/kivi-b019-objects GIT_ALTERNATE_OBJECT_DIRECTORIES=/opt/kivi-source/.git/objects /opt/kvbench/.venv/bin/python scripts/validate_kivi_b019_patch.py --device cpu --source-root /opt/kivi-source && /opt/kvbench/.venv/bin/python -m scripts.validate_kvquant_q23_long_context_patch --source-root /opt/kvquant-source && /opt/kvbench/.venv/bin/python -m scripts.phase12_unified_admission --run-campaign --stage "/home/rockrock/cmu_paper/'"$$stage_relative"'" --campaign-id "'"$$campaign_id"'" --git-sha "'"$$head"'"')"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		$(PHASE12_HOST_PYTHON) -m scripts.phase12_unified_admission --finalize-staged-campaign --stage "$$stage" --campaign-id "$$campaign_id"; \
		final_root="$$repository_root/artifacts/phase12/$$campaign_id"; \
		test -d "$$final_root" && test ! -L "$$final_root"; \
		test ! -e "$$stage" && test ! -L "$$stage"; \
		$(PHASE12_HOST_PYTHON) -m scripts.phase12_unified_admission --validate-campaign --artifact "$$final_root"; \
		printf 'PHASE12_CAMPAIGN_ID=%s\nPHASE12_CAMPAIGN_ARTIFACT=%s\n' "$$campaign_id" "$$final_root"; \
		preserve=0

validate-unified-admission:
	@test -n "$(PHASE12_CAMPAIGN_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE12_CAMPAIGN_ARTIFACT_required"}' >&2; exit 2; }
	@artifact="$$(realpath -e "$(PHASE12_CAMPAIGN_ARTIFACT)")"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test "$$repository_root" = "$(CURDIR)"; \
		test -d "$$artifact" && test ! -L "$(PHASE12_CAMPAIGN_ARTIFACT)"; \
		test "$$(dirname "$$artifact")" = "$$repository_root/artifacts/phase12"; \
		[[ "$$(basename "$$artifact")" =~ ^phase12-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}$$ ]]; \
		test -f docs/evidence/phase12/r2-publication.json && test ! -L docs/evidence/phase12/r2-publication.json; \
		test -f docs/evidence/phase12/unified-admission.json && test ! -L docs/evidence/phase12/unified-admission.json; \
		test -f docs/phase_reports/phase12-unified-admission.md && test ! -L docs/phase_reports/phase12-unified-admission.md; \
		$(PHASE12_HOST_PYTHON) -m scripts.phase12_unified_admission --validate-campaign --artifact "$$artifact"; \
		$(PHASE12_HOST_PYTHON) -m scripts.phase12_unified_admission --validate-final-evidence --artifact "$$artifact" --receipt-output docs/evidence/phase12/r2-publication.json --report-output docs/evidence/phase12/unified-admission.json --markdown-output docs/phase_reports/phase12-unified-admission.md

reference-kivi: validate-kivi-b019-patch
	@command -v docker >/dev/null
	@test "$$(docker image inspect kvbench-measurement:phase6a --format '{{.Id}}')" = "$(KIVI_REFERENCE_PARENT_CONFIG)"
	@task_root="$$(mktemp -d /tmp/kvbench-kivi-reference.XXXXXX)"; \
		trap 'rm -rf -- "$$task_root"' EXIT; \
		context="$$task_root/context"; \
		mkdir -p "$$context"; \
		cp --parents \
			docker/reference-kivi.Dockerfile \
			third_party/LOCK.json \
			third_party/patches/kivi/manifest.json \
			third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch \
			scripts/validate_kivi_b019_patch.py \
			reference/kivi/source_manifest.json \
			reference/kivi/generate_fixtures.py \
			reference/kivi/python-freeze.txt \
			"$$context"; \
		test "$$(find "$$context" -type f | wc -l)" = 8; \
		test ! -e "$$context/.env"; \
		DOCKER_BUILDKIT=1 docker build --pull=false --provenance=false --platform linux/amd64 \
			--build-arg KIVI_BUILD_REVISION="$(KIVI_REFERENCE_BUILD_REVISION)" \
			--iidfile "$$task_root/image.id" --metadata-file "$$task_root/metadata.json" \
			--tag "$(KIVI_REFERENCE_IMAGE)" --file "$$context/docker/reference-kivi.Dockerfile" "$$context"; \
		test "$$(tr -d '\r\n' < "$$task_root/image.id")" = "$(KIVI_REFERENCE_IMAGE_CONFIG)"; \
		test "$$($(PHASE2_PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["containerimage.digest"])' "$$task_root/metadata.json")" = "$(KIVI_REFERENCE_IMAGE_MANIFEST)"; \
		docker run --rm --gpus all "$(KIVI_REFERENCE_IMAGE)" kernel-probe >/dev/null; \
		docker run --rm --gpus all --entrypoint /usr/local/cuda-13.0/bin/compute-sanitizer "$(KIVI_REFERENCE_IMAGE)" --tool memcheck --error-exitcode=86 --target-processes all /opt/kvbench/.venv/bin/python /opt/kvbench-reference/reference/kivi/generate_fixtures.py sanitizer-probe; \
		dockerfile_sha="$$(sha256sum docker/reference-kivi.Dockerfile | cut -d ' ' -f 1)"; \
		source_sha="$$(sha256sum reference/kivi/source_manifest.json | cut -d ' ' -f 1)"; \
		build_sha="$$(sha256sum reference/kivi/build_manifest.json | cut -d ' ' -f 1)"; \
		docker run --rm --gpus all --user "$$(id -u):$$(id -g)" --tmpfs /tmp:rw,exec,nosuid,size=1g --mount type=bind,src="$(CURDIR)/reference/kivi",dst=/output "$(KIVI_REFERENCE_IMAGE)" fixtures --output /output/fixtures --image-config-digest "$(KIVI_REFERENCE_IMAGE_CONFIG)" --dockerfile-sha256 "$$dockerfile_sha" --source-manifest-sha256 "$$source_sha" --build-manifest-sha256 "$$build_sha" --extension-sha256 "$(KIVI_REFERENCE_EXTENSION_SHA256)"
	@$(PHASE3_ENV) $(PHASE3_PYTHON) reference/kivi/validate_fixtures.py --image "$(KIVI_REFERENCE_IMAGE)"

validate-reference-kivi: validate-kivi-b019-patch
	@$(PHASE3_ENV) $(PHASE3_PYTHON) reference/kivi/validate_fixtures.py --image "$(KIVI_REFERENCE_IMAGE)"

phase6a-source-safety:
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@if git ls-files --error-unmatch .env >/dev/null 2>&1; then echo '{"status":"FAIL","reason":"env_tracked"}' >&2; exit 2; fi
	@git check-ignore -q .env || { echo '{"status":"FAIL","reason":"env_not_ignored"}' >&2; exit 2; }
	@if test -e .env || test -L .env; then \
		test -f .env && test ! -L .env || { echo '{"status":"FAIL","reason":"env_unsafe"}' >&2; exit 2; }; \
		mode="$$(stat -c '%a' .env)"; (( (8#$$mode & 077) == 0 )) || { echo '{"status":"FAIL","reason":"env_permissions_too_broad"}' >&2; exit 2; }; \
	fi
	@grep -Fxq '.env' .dockerignore && grep -Fxq '.env.*' .dockerignore || { echo '{"status":"FAIL","reason":"env_not_excluded_from_docker"}' >&2; exit 2; }

measurement-container: phase6a-source-safety
	@command -v docker >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_cli_unavailable"}' >&2; exit 2; }
	@docker info >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_daemon_unavailable"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-build.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		mkdir "$$task_root/context"; \
		head="$$(git rev-parse HEAD)"; \
		git archive --format=tar "$$head" | tar -xf - -C "$$task_root/context"; \
		test ! -e "$$task_root/context/.env"; \
		DOCKER_BUILDKIT=1 docker build --pull --platform linux/amd64 \
			--iidfile "$$task_root/image.id" \
			--label "org.opencontainers.image.revision=$$head" \
			--tag "$(MEASUREMENT_IMAGE)" \
			--file "$$task_root/context/docker/measurement.Dockerfile" \
			"$$task_root/context"; \
		image_id="$$(tr -d '\r\n' < "$$task_root/image.id")"; \
		[[ "$$image_id" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"FAIL","reason":"invalid_image_config_digest"}' >&2; exit 2; }; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,validate_container_image_inspect; raw=Path(sys.argv[1]).read_bytes(); _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=sys.argv[4],secret_environment=os.environ); history=Path(sys.argv[5]).read_bytes(); raise SystemExit(0 if result["status"] == "PASS" and not configured_secret_value_names(history,os.environ) else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$head" "$$task_root/image-history.jsonl" || { echo '{"status":"FAIL","reason":"image_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		docker image save --output "$$task_root/image-save.tar" "$$image_id"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import validate_docker_image_save_archive; result=validate_docker_image_save_archive(Path(sys.argv[1]),secret_environment=os.environ); raise SystemExit(0 if result["status"] == "PASS" else 2)' "$$task_root/image-save.tar" || { echo '{"status":"FAIL","reason":"image_layer_secret_validation_failed"}' >&2; exit 2; }; \
		printf '{"status":"BUILT_UNCERTIFIED","image_reference":"%s","image_config_digest":"%s","base_image_digest":"%s"}\n' "$(MEASUREMENT_IMAGE)" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)"

observe-measurement-container-lock: phase6a-source-safety
	@[[ "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"BLOCKED","reason":"MEASUREMENT_IMAGE_CONFIG_DIGEST_required"}' >&2; exit 2; }
	@test ! -e preflight/measurement-container-system-packages.lock.json && test ! -L preflight/measurement-container-system-packages.lock.json || { echo '{"status":"BLOCKED","reason":"observed_container_system_lock_exists"}' >&2; exit 2; }
	@test ! -e preflight/measurement-container-system-packages.expected.json && test ! -L preflight/measurement-container-system-packages.expected.json || { echo '{"status":"BLOCKED","reason":"expected_container_system_lock_exists"}' >&2; exit 2; }
	@command -v docker >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_cli_unavailable"}' >&2; exit 2; }
	@docker info >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_daemon_unavailable"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-lock.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		mkdir "$$task_root/source"; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git archive --format=tar "$$head" | tar -xf - -C "$$task_root/source"; \
		test ! -e "$$task_root/source/.env"; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,validate_container_image_inspect; _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=sys.argv[4],secret_environment=os.environ); history=Path(sys.argv[5]).read_bytes(); raise SystemExit(0 if result["status"] == "PASS" and not configured_secret_value_names(history,os.environ) else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$head" "$$task_root/image-history.jsonl" || { echo '{"status":"FAIL","reason":"candidate_image_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		chmod 0444 "$$task_root/image-inspect.json"; \
		docker run --rm --read-only --network=none \
			--tmpfs /tmp:rw,nosuid,nodev,size=1g \
			--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
			--mount "type=bind,src=$$task_root/image-inspect.json,dst=/run/kvbench/image-inspect.json,readonly" \
			--workdir /workspace \
			--entrypoint /opt/kvbench/.venv/bin/python "$$image_id" \
			-m preflight.run_preflight \
			--observe-measurement-container-system-lock \
			--container-image-reference "$(MEASUREMENT_IMAGE)" \
			--container-image-config-digest "$$image_id" \
			--container-base-image-digest "$(MEASUREMENT_BASE_IMAGE_DIGEST)" \
			--container-image-inspect-json /run/kvbench/image-inspect.json \
			--container-source-revision "$$head" \
			> "$$task_root/observed.json"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,json_bytes,load_measurement_container_system_lock,publish_atomic_exclusive; source=Path(sys.argv[1]); payload,result=load_measurement_container_system_lock(source,expected_image_config_digest=sys.argv[3],expected_source_revision=sys.argv[4]); raw=json_bytes(payload) if payload is not None else b""; valid=payload is not None and result["status"] == "PASS" and not configured_secret_value_names(raw,os.environ); valid and publish_atomic_exclusive(Path(sys.argv[2]),raw); raise SystemExit(0 if valid else 2)' "$$task_root/observed.json" "preflight/measurement-container-system-packages.lock.json" "$$image_id" "$$head" || { echo '{"status":"FAIL","reason":"observed_container_system_lock_invalid"}' >&2; exit 2; }; \
		printf '{"status":"OBSERVED_UNTRUSTED_REVIEW_REQUIRED","path":"preflight/measurement-container-system-packages.lock.json","candidate_image_config_digest":"%s","candidate_source_revision":"%s"}\n' "$$image_id" "$$head"

measurement-container-lock-review: phase6a-source-safety
	@for lock_path in preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json; do \
		test -f "$$lock_path" && test ! -L "$$lock_path" && git ls-files --error-unmatch "$$lock_path" >/dev/null || { echo '{"status":"BLOCKED","reason":"reviewed_container_system_locks_absent_or_untracked"}' >&2; exit 2; }; \
	done
	@cmp -s preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json || { echo '{"status":"BLOCKED","reason":"reviewed_container_system_locks_differ"}' >&2; exit 2; }
	@PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,load_measurement_container_system_lock; paths=[Path("preflight/measurement-container-system-packages.lock.json"),Path("preflight/measurement-container-system-packages.expected.json")]; loaded=[load_measurement_container_system_lock(path) for path in paths]; raw=[path.read_bytes() for path in paths]; valid=all(payload is not None and result["status"] == "PASS" for payload,result in loaded) and raw[0] == raw[1] and all(not configured_secret_value_names(item,os.environ) for item in raw); raise SystemExit(0 if valid else 2)' || { echo '{"status":"FAIL","reason":"reviewed_container_system_lock_invalid"}' >&2; exit 2; }
	@printf '{"status":"REVIEWED_LOCK_BYTES_EQUAL","authority":"rebuild_and_exact_runtime_verification_still_required"}\n'

verify-measurement-container: measurement-container-lock-review
	@PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_measurement_container -v
	@[[ "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"BLOCKED","reason":"MEASUREMENT_IMAGE_CONFIG_DIGEST_required"}' >&2; exit 2; }
	@command -v docker >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_cli_unavailable"}' >&2; exit 2; }
	@docker info >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_daemon_unavailable"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-verify.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,re,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,validate_container_image_inspect; _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=None,secret_environment=os.environ); history=Path(sys.argv[4]).read_bytes(); revision=result.get("source_revision"); valid=result["status"] == "PASS" and isinstance(revision,str) and re.fullmatch(r"[0-9a-f]{40}",revision) is not None and not configured_secret_value_names(history,os.environ); valid and Path(sys.argv[5]).write_text(revision+"\n",encoding="ascii"); raise SystemExit(0 if valid else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$task_root/image-history.jsonl" "$$task_root/image-revision" || { echo '{"status":"FAIL","reason":"image_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		image_revision="$$(tr -d '\r\n' < "$$task_root/image-revision")"; \
		git cat-file -e "$$image_revision^{commit}"; \
		git merge-base --is-ancestor "$$image_revision" "$$head"; \
		git diff --quiet "$$image_revision" "$$head" -- docker/measurement.Dockerfile preflight/requirements-e00.txt preflight/requirements-phase3.txt; \
		for lock_path in preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json; do \
			git cat-file -e "$$image_revision:$$lock_path"; \
			test "$$(git rev-parse "$$image_revision:$$lock_path")" = "$$(git rev-parse "HEAD:$$lock_path")"; \
		done; \
		docker image save --output "$$task_root/image-save.tar" "$$image_id"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import validate_docker_image_save_archive; result=validate_docker_image_save_archive(Path(sys.argv[1]),secret_environment=os.environ); raise SystemExit(0 if result["status"] == "PASS" else 2)' "$$task_root/image-save.tar" || { echo '{"status":"FAIL","reason":"image_layer_secret_validation_failed"}' >&2; exit 2; }; \
		docker run --rm --network=none --gpus "device=$(MEASUREMENT_GPU_UUID)" --entrypoint /usr/bin/nvidia-smi "$$image_id" --query-gpu=uuid,name,compute_cap --format=csv,noheader,nounits > "$$task_root/gpu.csv"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import csv,sys; from pathlib import Path; rows=list(csv.reader(Path(sys.argv[1]).read_text().splitlines())); expected=[sys.argv[2],"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","12.0"]; raise SystemExit(0 if len(rows) == 1 and [item.strip() for item in rows[0]] == expected else 2)' "$$task_root/gpu.csv" "$(MEASUREMENT_GPU_UUID)" || { echo '{"status":"FAIL","reason":"exact_gpu_runtime_probe_failed"}' >&2; exit 2; }; \
		printf '{"status":"IDENTITY_AND_RUNTIME_READY_NOT_CERTIFIED","image_config_digest":"%s","base_image_digest":"%s"}\n' "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)"

preflight-container: verify-measurement-container
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-container.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/source/.env"; \
		mkdir -p "$$task_root/source/artifacts/phase6a/container_g0"; \
		mkdir -p "$(CURDIR)/artifacts/phase6a/container_g0"; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import re,sys; from pathlib import Path; from preflight.run_preflight import validate_container_image_inspect; _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=None); revision=result.get("source_revision"); valid=result["status"] == "PASS" and isinstance(revision,str) and re.fullmatch(r"[0-9a-f]{40}",revision) is not None; valid and Path(sys.argv[4]).write_text(revision+"\n",encoding="ascii"); raise SystemExit(0 if valid else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$task_root/image-revision"; \
		image_revision="$$(tr -d '\r\n' < "$$task_root/image-revision")"; \
		git cat-file -e "$$image_revision^{commit}"; \
		git merge-base --is-ancestor "$$image_revision" "$$head"; \
		git diff --quiet "$$image_revision" "$$head" -- docker/measurement.Dockerfile preflight/requirements-e00.txt preflight/requirements-phase3.txt; \
		for lock_path in preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json; do \
			git cat-file -e "$$image_revision:$$lock_path"; \
			test "$$(git rev-parse "$$image_revision:$$lock_path")" = "$$(git rev-parse "HEAD:$$lock_path")"; \
		done; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		docker image save --output "$$task_root/image-save.tar" "$$image_id"; \
		scan_status=0; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,json_bytes,sha256_bytes,validate_docker_image_save_archive; history=Path(sys.argv[1]).read_bytes(); history_secrets=configured_secret_value_names(history,os.environ); layers=validate_docker_image_save_archive(Path(sys.argv[2]),secret_environment=os.environ); record={"schema_version":"kvbench-measurement-container-build-scan-1.0.0","image_config_digest":sys.argv[3],"image_history":{"sha256":sha256_bytes(history),"size_bytes":len(history),"line_count":len(history.splitlines()),"configured_secret_variables":history_secrets},"image_layers":layers}; Path(sys.argv[4]).write_bytes(json_bytes(record)); valid=layers["status"] == "PASS" and not history_secrets and layers["model_weight_path_count"] == 0; raise SystemExit(0 if valid else 2)' "$$task_root/image-history.jsonl" "$$task_root/image-save.tar" "$$image_id" "$$task_root/image-build-scan.json" || scan_status=$$?; \
		rm -f -- "$$task_root/image-save.tar"; \
		if (( scan_status != 0 )); then rm -f -- "$$task_root/image-history.jsonl"; echo '{"status":"FAIL","reason":"image_build_scan_failed"}' >&2; exit 2; fi; \
		touch "$$task_root/runtime-inspect.json"; \
		chmod 0444 "$$task_root/image-inspect.json" "$$task_root/image-history.jsonl" "$$task_root/image-build-scan.json"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,nosuid,nodev,size=8g \
			--tmpfs /root:rw,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase6a/container_g0,dst=/workspace/artifacts/phase6a/container_g0" \
			--mount "type=bind,src=$$task_root/image-inspect.json,dst=/run/kvbench/image-inspect.json,readonly" \
			--mount "type=bind,src=$$task_root/runtime-inspect.json,dst=/run/kvbench/runtime-inspect.json,readonly" \
			--mount "type=bind,src=$$task_root/image-history.jsonl,dst=/run/kvbench/image-history.jsonl,readonly" \
			--mount "type=bind,src=$$task_root/image-build-scan.json,dst=/run/kvbench/image-build-scan.json,readonly" \
			--workdir /workspace "$$image_id" \
			/opt/kvbench/.venv/bin/python preflight/run_preflight.py \
			--measurement-container \
			--container-image-reference "$(MEASUREMENT_IMAGE)" \
			--container-image-config-digest "$$image_id" \
			--container-base-image-digest "$(MEASUREMENT_BASE_IMAGE_DIGEST)" \
			--container-image-inspect-json /run/kvbench/image-inspect.json \
			--container-runtime-inspect-json /run/kvbench/runtime-inspect.json \
			--container-image-history-jsonl /run/kvbench/image-history.jsonl \
			--container-image-build-scan-json /run/kvbench/image-build-scan.json \
			--container-source-revision "$$image_revision")"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]] || { echo '{"status":"FAIL","reason":"invalid_runtime_container_id"}' >&2; exit 2; }; \
		chmod 0644 "$$task_root/runtime-inspect.json"; \
		docker container inspect "$$cid" > "$$task_root/runtime-inspect.json"; \
		chmod 0444 "$$task_root/runtime-inspect.json"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import validate_container_runtime_inspect; _,result=validate_container_runtime_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_hostname=sys.argv[3][:12],secret_environment=os.environ); raise SystemExit(0 if result["status"] == "PASS" else 2)' "$$task_root/runtime-inspect.json" "$$image_id" "$$cid" || { echo '{"status":"FAIL","reason":"runtime_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		docker start --attach "$$cid"; \
		preserve=0

phase6a-bf16-container-parity: verify-measurement-container
	@[[ "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"BLOCKED","reason":"MEASUREMENT_IMAGE_CONFIG_DIGEST_required"}' >&2; exit 2; }
	@test -n "$(CONTAINER_G0_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"CONTAINER_G0_ARTIFACT_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-parity.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_PARITY_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/source/.env"; \
		mkdir -p "$$task_root/source/artifacts/phase6a/bf16_parity"; \
		mkdir -p "$(CURDIR)/artifacts/phase6a/bf16_parity"; \
		g0="$$(realpath "$(CONTAINER_G0_ARTIFACT)")"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		model_snapshot="$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		test -d "$$g0" && test -d "$$model_root" && test -d "$$model_snapshot"; \
		overall=0; \
		for graph_mode in eager cuda_graph; do \
			cid="$$(docker create --read-only --network=none --pid=host \
				--gpus "device=$(MEASUREMENT_GPU_UUID)" \
				--tmpfs /tmp:rw,nosuid,nodev,size=8g \
				--tmpfs /root:rw,nosuid,nodev,size=2g \
				--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
				--mount "type=bind,src=$$task_root/source,dst=/home/rockrock/cmu_paper,readonly" \
				--mount "type=bind,src=$(CURDIR)/artifacts/phase6a/bf16_parity,dst=/workspace/artifacts/phase6a/bf16_parity" \
				--mount "type=bind,src=$$g0,dst=/run/kvbench/container-g0,readonly" \
				--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
				--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/workspace/src:/workspace \
				--workdir /workspace "$$image_id" \
				/opt/kvbench/.venv/bin/python scripts/phase6a_bf16_parity.py \
				--graph-mode "$$graph_mode" \
				--image-reference "$(MEASUREMENT_IMAGE)" \
				--image-config-digest "$$image_id" \
				--container-g0-artifact /run/kvbench/container-g0)"; \
			[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
			if ! docker start --attach "$$cid"; then overall=1; fi; \
			docker rm -f "$$cid" >/dev/null; cid=""; \
		done; \
		(( overall == 0 )) || exit 1; \
		preserve=0

remediate-b018-turboquant: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE6_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6-b018.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_B018_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		mkdir -p "$$task_root/source/artifacts/phase6" "$(CURDIR)/artifacts/phase6"; \
		test ! -e "$$task_root/source/.env"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,nosuid,nodev,size=8g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/source,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase6,dst=/home/rockrock/cmu_paper/artifacts/phase6" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env HF_HUB_OFFLINE=1 \
			--env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 \
			--env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TRITON_CACHE_DIR=/root/.triton \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /opt/kvbench/.venv/bin/python "$$image_id" \
			-m scripts.phase6_turboquant_admission \
			--b018-sanitizer-only)"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		preserve=0

admit-turboquant: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE6_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6-admission.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_PHASE6_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		chmod -R a-w "$$task_root/source/docs/evidence/e00"; \
		test -z "$$(find "$$task_root/source/docs/evidence/e00" -perm /222 -print -quit)"; \
		mkdir "$$task_root/source/.venv"; \
		ln -s /opt/kvbench/.venv/bin "$$task_root/source/.venv/bin"; \
		ln -s /opt/kvbench/.venv/lib "$$task_root/source/.venv/lib"; \
		ln -s /opt/kvbench/.venv/pyvenv.cfg "$$task_root/source/.venv/pyvenv.cfg"; \
		mkdir "$$task_root/source/.phase3"; \
		ln -s /opt/kvbench/.phase3/site-packages "$$task_root/source/.phase3/site-packages"; \
		mkdir -p "$$task_root/source/artifacts/phase6" "$(CURDIR)/artifacts/phase6"; \
		test ! -e "$$task_root/source/.env"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		model_snapshot="$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		test -d "$$model_root" && test -d "$$model_snapshot"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,nosuid,nodev,size=8g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
			--mount "type=bind,src=$$task_root/source,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase6,dst=/home/rockrock/cmu_paper/artifacts/phase6" \
			--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env HF_HUB_OFFLINE=1 \
			--env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 \
			--env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TRITON_CACHE_DIR=/root/.triton \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /usr/bin/bash "$$image_id" \
			--noprofile --norc -eu -o pipefail -c \
			'make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-cuda && make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-graph && /opt/kvbench/.venv/bin/python scripts/phase6_turboquant_admission.py')"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		preserve=0

validate-admission-turboquant:
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase6_turboquant_admission --validate-only

phase6-r2-outer-bundle:
	@test -n "$(PHASE6_R2_OUTER_RUN_ID)" || { echo '{"status":"FAIL","reason":"PHASE6_R2_OUTER_RUN_ID_required"}' >&2; exit 2; }
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase6_r2_outer_bundle build --run-id "$(PHASE6_R2_OUTER_RUN_ID)"

validate-phase6-r2-outer-bundle:
	@test -n "$(PHASE6_R2_OUTER_ARTIFACT)" || { echo '{"status":"FAIL","reason":"PHASE6_R2_OUTER_ARTIFACT_required"}' >&2; exit 2; }
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase6_r2_outer_bundle validate "$(PHASE6_R2_OUTER_ARTIFACT)"

admit-kivi: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST)
admit-kivi: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase8_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase8-kivi-admission.XXXXXX)"; \
		cid=""; reference_cid=""; preserve=1; \
		cleanup() { \
			if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$reference_cid" ]]; then docker rm -f "$$reference_cid" >/dev/null 2>&1 || true; fi; \
			if (( preserve == 0 )); then \
				chmod -R u+w "$$task_root" 2>/dev/null || true; \
				rm -rf -- "$$task_root"; \
			else \
				printf '{"status":"FAILED_PHASE8_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; \
			fi; \
		}; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST)"; \
		reference_image="$(KIVI_REFERENCE_IMAGE)@$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$image_id" --format '{{.Id}}')" = "$$image_id"; \
		test "$$(docker image inspect "$$reference_image" --format '{{.Id}}')" = "$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$reference_image" --format '{{index .Config.Labels "org.kvbench.reference.parent.config_digest"}}')" = "$$image_id"; \
		test "$$($(PHASE2_PYTHON) -c 'import json; print(json.load(open("reference/kivi/build_manifest.json", encoding="utf-8"))["image"]["config_digest"])')" = "$(PHASE8_KIVI_REFERENCE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/source/.env"; \
		chmod -R a-w "$$task_root/source/docs/evidence/e00"; \
		mkdir "$$task_root/source/.venv"; \
		ln -s /opt/kvbench/.venv/bin "$$task_root/source/.venv/bin"; \
		ln -s /opt/kvbench/.venv/lib "$$task_root/source/.venv/lib"; \
		ln -s /opt/kvbench/.venv/pyvenv.cfg "$$task_root/source/.venv/pyvenv.cfg"; \
		mkdir "$$task_root/source/.phase3"; \
		ln -s /opt/kvbench/.phase3/site-packages "$$task_root/source/.phase3/site-packages"; \
		mkdir -p "$$task_root/source/artifacts/phase8" "$(CURDIR)/artifacts/phase8"; \
		mkdir "$$task_root/kivi-source" "$$task_root/kivi-extension"; \
		reference_cid="$$(docker create --network=none "$$reference_image")"; \
		[[ "$$reference_cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker cp "$$reference_cid:/opt/kivi-source/." "$$task_root/kivi-source"; \
		docker cp "$$reference_cid:/opt/kivi-source/quant/kivi_gemv.cpython-312-x86_64-linux-gnu.so" "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so"; \
		docker rm -f "$$reference_cid" >/dev/null; reference_cid=""; \
		test -f "$$task_root/kivi-source/models/kivi_gqa.py" && test ! -L "$$task_root/kivi-source/models/kivi_gqa.py"; \
		cp --preserve=mode,timestamps "$$task_root/kivi-source/models/kivi_gqa.py" "$$task_root/kivi-gqa.authority.py"; \
		git -C "$$task_root/kivi-source" clean -fdx; \
		cp --preserve=mode,timestamps "$$task_root/kivi-gqa.authority.py" "$$task_root/kivi-source/models/kivi_gqa.py"; \
		test "$$(sha256sum "$$task_root/kivi-source/quant/new_pack.py" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_NEW_PACK_SHA256)"; \
		test "$$(sha256sum "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_EXTENSION_SHA256)"; \
		$(PHASE3_ENV) $(PHASE3_PYTHON) scripts/validate_kivi_b019_patch.py \
			--device cpu --source-root "$$task_root/kivi-source" \
			> "$$task_root/kivi-source-validation.json"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		model_snapshot="$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		test -d "$$model_root" && test -d "$$model_snapshot"; \
		artifact_root="$$(realpath "$(CURDIR)/artifacts/phase8")"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=8g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/source,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$artifact_root,dst=/home/rockrock/cmu_paper/artifacts/phase8" \
			--mount "type=bind,src=$$task_root/kivi-source,dst=/opt/kivi-source,readonly" \
			--mount "type=bind,src=$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so,dst=/opt/kvbench/.phase3/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so,readonly" \
			--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env HF_HUB_OFFLINE=1 \
			--env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 \
			--env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TRITON_CACHE_DIR=/root/.triton \
			--env KVBENCH_KIVI_SOURCE_ROOT=/opt/kivi-source \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /usr/bin/bash "$$image_id" \
			--noprofile --norc -eu -o pipefail -c \
			'/usr/bin/mkdir -p /tmp/kivi-b019-objects && GIT_OBJECT_DIRECTORY=/tmp/kivi-b019-objects GIT_ALTERNATE_OBJECT_DIRECTORIES=/opt/kivi-source/.git/objects /opt/kvbench/.venv/bin/python scripts/validate_kivi_b019_patch.py --device cpu --source-root /opt/kivi-source && make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-cuda && make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-graph && /opt/kvbench/.venv/bin/python scripts/phase8_kivi_admission.py')"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		preserve=0

validate-admission-kivi:
	@test -n "$(PHASE8_KIVI_ADMISSION_ARTIFACT)" || { echo '{"status":"FAIL","reason":"PHASE8_KIVI_ADMISSION_ARTIFACT_required"}' >&2; exit 2; }
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase8_kivi_admission --validate-only --artifact "$(PHASE8_KIVI_ADMISSION_ARTIFACT)"

phase8-r2-outer-bundle:
	@test -n "$(PHASE8_R2_INNER_ARTIFACT)" || { echo '{"status":"FAIL","reason":"PHASE8_R2_INNER_ARTIFACT_required"}' >&2; exit 2; }
	@test -n "$(PHASE8_R2_OUTER_RUN_ID)" || { echo '{"status":"FAIL","reason":"PHASE8_R2_OUTER_RUN_ID_required"}' >&2; exit 2; }
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase8_r2_outer_bundle build --source-bundle "$(PHASE8_R2_INNER_ARTIFACT)" --run-id "$(PHASE8_R2_OUTER_RUN_ID)"

validate-phase8-r2-outer-bundle:
	@test -n "$(PHASE8_R2_INNER_ARTIFACT)" || { echo '{"status":"FAIL","reason":"PHASE8_R2_INNER_ARTIFACT_required"}' >&2; exit 2; }
	@test -n "$(PHASE8_R2_OUTER_ARTIFACT)" || { echo '{"status":"FAIL","reason":"PHASE8_R2_OUTER_ARTIFACT_required"}' >&2; exit 2; }
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase8_r2_outer_bundle validate "$(PHASE8_R2_OUTER_ARTIFACT)" --source-bundle "$(PHASE8_R2_INNER_ARTIFACT)"

validate-phase8-r2-outer-publication:
	@test -n "$(PHASE8_R2_INNER_ARTIFACT)" || { echo '{"status":"FAIL","reason":"PHASE8_R2_INNER_ARTIFACT_required"}' >&2; exit 2; }
	@test -n "$(PHASE8_R2_OUTER_ARTIFACT)" || { echo '{"status":"FAIL","reason":"PHASE8_R2_OUTER_ARTIFACT_required"}' >&2; exit 2; }
	@test -f "$(PHASE8_R2_OUTER_RECEIPT)" || { echo '{"status":"FAIL","reason":"PHASE8_R2_OUTER_RECEIPT_absent"}' >&2; exit 2; }
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase8_r2_outer_bundle validate-publication "$(PHASE8_R2_OUTER_ARTIFACT)" --source-bundle "$(PHASE8_R2_INNER_ARTIFACT)" --receipt "$(PHASE8_R2_OUTER_RECEIPT)"

publish-artifact-r2:
	@test -n "$(ARTIFACT)" || { echo '{"status":"FAIL","reason":"ARTIFACT_required"}' >&2; exit 2; }
	@$(R2_ARTIFACT) publish "$(ARTIFACT)"

verify-artifact-r2:
	@test -n "$(ROOT_SHA256)" || { echo '{"status":"FAIL","reason":"ROOT_SHA256_required"}' >&2; exit 2; }
	@[[ "$(ROOT_SHA256)" =~ ^[0-9a-f]{64}$$ ]] || { echo '{"status":"FAIL","reason":"ROOT_SHA256_invalid"}' >&2; exit 2; }
	@$(R2_ARTIFACT) verify "$(ROOT_SHA256)"

smoke: export KVBENCH_METHOD := $(METHOD)
smoke:
	@$(PHASE2_VALIDATE) method
	@$(PHASE2_CLI) run --plan configs/plans/smoke.yaml --dry-run
	@echo '{"status":"validation_only","target":"smoke","requested_method":"$(METHOD)","plan_scope":"all_methods","timing_collected":false}'

pilot: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)
pilot: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase13_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"clean_committed_phase13_tree_required"}' >&2; exit 2; }
	@$(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --validate-phase13b-entry
	@task_root="$$(mktemp -d /tmp/kvbench-phase13-pilot.XXXXXX)"; \
		cid=""; reference_cid=""; analysis_cid=""; stage=""; campaign_id=""; final_root=""; head=""; prefix_campaign_root=""; preserve=1; \
		cleanup() { \
			status=$$?; \
			if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$reference_cid" ]]; then docker rm -f "$$reference_cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$analysis_cid" ]]; then docker rm -f "$$analysis_cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$stage" && -d "$$stage" ]]; then printf '{"status":"PHASE13_STAGING_EVIDENCE_PRESERVED","path":"%s"}\n' "$$stage" >&2; fi; \
			if (( preserve == 0 )); then \
				chmod -R u+w "$$task_root" 2>/dev/null || true; \
				rm -rf -- "$$task_root"; \
			else \
				printf '{"status":"PHASE13_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; \
			fi; \
			trap - EXIT; \
			exit $$status; \
		}; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test "$$repository_root" = "$(CURDIR)"; \
		git cat-file -e "$$head:docs/plans/phase13-pilot-scan.md"; \
		git cat-file -e "$$head:docs/plans/phase13-pilot-execution-order.json"; \
		image_id="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)"; \
		analysis_image="$(PHASE13_ANALYSIS_IMAGE_CONFIG_DIGEST)"; \
		test "$$(docker image inspect "$$image_id" --format '{{.Id}}')" = "$$image_id"; \
		test "$$(docker image inspect "$$analysis_image" --format '{{.Id}}')" = "$$analysis_image"; \
		campaign_id="$$($(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --new-campaign-id --git-sha "$$head")"; \
		[[ "$$campaign_id" =~ ^phase13-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}$$ ]]; \
		prefix_seed_root="$(PHASE13_PREFIX_SEED_ROOT)"; \
		test -d "$$prefix_seed_root" && test ! -L "$$prefix_seed_root"; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --validate-prefix-seed --seed-root "$$prefix_seed_root"; \
		prefix_campaign_root="$$repository_root/artifacts/phase13_prefix_catalogs/$$campaign_id"; \
		test ! -e "$$prefix_campaign_root" && test ! -L "$$prefix_campaign_root"; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --materialize-prefix-seed --seed-root "$$prefix_seed_root" --destination "$$prefix_campaign_root" --git-sha "$$head"; \
		stage="$$($(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --reserve-campaign --campaign-id "$$campaign_id" --git-sha "$$head")"; \
		test -d "$$stage" && test ! -L "$$stage"; \
		test "$$(realpath -e "$$stage")" = "$$stage"; \
		case "$$stage" in "$$repository_root"/artifacts/phase13/.kvbench-staging/"$$campaign_id".*.staging) ;; *) echo '{"status":"BLOCKED","reason":"unsafe_phase13_stage"}' >&2; exit 2;; esac; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/repository" remote remove origin; \
		test "$$(git -C "$$task_root/repository" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env" && test ! -L "$$task_root/repository/.env"; \
		mkdir "$$task_root/repository/.venv" "$$task_root/repository/.phase3"; \
		ln -s /opt/kvbench/.venv/bin "$$task_root/repository/.venv/bin"; \
		ln -s /opt/kvbench/.venv/lib "$$task_root/repository/.venv/lib"; \
		ln -s /opt/kvbench/.venv/pyvenv.cfg "$$task_root/repository/.venv/pyvenv.cfg"; \
		ln -s /opt/kvbench/.phase3/site-packages "$$task_root/repository/.phase3/site-packages"; \
		stage_relative="$${stage#$$repository_root/}"; \
		test "$$stage_relative" != "$$stage"; \
		mkdir -p "$$task_root/repository/$$(dirname "$$stage_relative")"; \
		mkdir "$$task_root/repository/$$stage_relative"; \
		for immutable_relative in docs/evidence/e00 reference/kvquant/fixtures reference/kvquant_phase11pr/fixtures; do \
			immutable_root="$$task_root/repository/$$immutable_relative"; \
			test -d "$$immutable_root" && test ! -L "$$immutable_root"; \
			chmod -R a-w -- "$$immutable_root"; \
			test -z "$$(find "$$immutable_root" -perm /222 -print -quit)"; \
		done; \
		phase12_artifact_root="$$repository_root/artifacts/phase12"; \
		test -d "$$phase12_artifact_root" && test ! -L "$$phase12_artifact_root"; \
		test "$$(realpath -e "$$phase12_artifact_root")" = "$$phase12_artifact_root"; \
		mkdir -p "$$task_root/repository/artifacts/phase12"; \
		phase13b_bundle="$(PHASE13B_LOCAL_BUNDLE)"; \
		test -d "$$phase13b_bundle" && test ! -L "$$phase13b_bundle"; \
		test "$$(realpath -e "$$phase13b_bundle")" = "$$phase13b_bundle"; \
		test "$$($(PHASE13_HOST_PYTHON) -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1], environ={}).root_sha256)' "$$phase13b_bundle")" = "$(PHASE13B_LOCAL_ROOT_SHA256)"; \
		mkdir -p "$$task_root/repository/artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission"; \
		phase13rq4_bundle="$(PHASE13RQ4_LOCAL_BUNDLE)"; \
		test -d "$$phase13rq4_bundle" && test ! -L "$$phase13rq4_bundle"; \
		test "$$(realpath -e "$$phase13rq4_bundle")" = "$$phase13rq4_bundle"; \
		test "$$($(PHASE13_HOST_PYTHON) -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1], environ={}).root_sha256)' "$$phase13rq4_bundle")" = "$(PHASE13RQ4_LOCAL_ROOT_SHA256)"; \
		mkdir -p "$$task_root/repository/artifacts/phase13rq4/phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd"; \
		reference_image="$(KIVI_REFERENCE_IMAGE)@$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$reference_image" --format '{{.Id}}')" = "$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$reference_image" --format '{{index .Config.Labels "org.kvbench.reference.parent.config_digest"}}')" = "$$image_id"; \
		mkdir "$$task_root/kivi-source" "$$task_root/kivi-extension"; \
		reference_cid="$$(docker create --network=none "$$reference_image")"; \
		[[ "$$reference_cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker cp "$$reference_cid:/opt/kivi-source/." "$$task_root/kivi-source"; \
		docker cp "$$reference_cid:/opt/kivi-source/quant/kivi_gemv.cpython-312-x86_64-linux-gnu.so" "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so"; \
		docker rm -f "$$reference_cid" >/dev/null; reference_cid=""; \
		cp --preserve=mode,timestamps "$$task_root/kivi-source/models/kivi_gqa.py" "$$task_root/kivi-gqa.authority.py"; \
		git -C "$$task_root/kivi-source" clean -fdx; \
		cp --preserve=mode,timestamps "$$task_root/kivi-gqa.authority.py" "$$task_root/kivi-source/models/kivi_gqa.py"; \
		test "$$(sha256sum "$$task_root/kivi-source/quant/new_pack.py" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_NEW_PACK_SHA256)"; \
		test "$$(sha256sum "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_EXTENSION_SHA256)"; \
		$(PHASE13_HOST_PYTHON) scripts/validate_kivi_b019_patch.py --device cpu --source-root "$$task_root/kivi-source" > "$$task_root/kivi-source-validation.json"; \
		git clone --quiet --no-local --no-checkout "$(PHASE11DQ23_KVQUANT_SOURCE_ROOT)" "$$task_root/kvquant-source"; \
		git -C "$$task_root/kvquant-source" checkout --quiet --detach "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		git -C "$$task_root/kvquant-source" remote remove origin; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD)" = "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD^{tree})" = "$(PHASE11DQ23_KVQUANT_TREE)"; \
		test -z "$$(git -C "$$task_root/kvquant-source" status --porcelain=v1 --untracked-files=all)"; \
		$(PHASE13_HOST_PYTHON) -m scripts.validate_kvquant_q23_long_context_patch --source-root "$$task_root/kvquant-source" > "$$task_root/kvquant-source-validation.json"; \
		test "$$($(PHASE2_PYTHON) -I -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["aggregate_patch_sha256"])' "$$task_root/kvquant-source-validation.json")" = "$(PHASE11DQ23_KVQUANT_PATCH_SHA256)"; \
		mkdir "$$task_root/kvquant-build"; \
		for file in setup_cuda.py quant_cuda.cpp quant_cuda_kernel.cu measurement_cuda_kernel.cu; do \
			relative="deployment/kvquant/$$file"; source="$$task_root/kvquant-source/$$relative"; \
			committed_sha256="$$(git -C "$$task_root/kvquant-source" cat-file blob "$(PHASE11DQ23_KVQUANT_COMMIT):$$relative" | sha256sum | cut -d ' ' -f 1)"; \
			test "$$(sha256sum "$$source" | cut -d ' ' -f 1)" = "$$committed_sha256"; \
			cp -- "$$source" "$$task_root/kvquant-build/$$file"; \
		done; \
		calibration_root="$(PHASE11_KVQUANT_CALIBRATION)"; \
		test -d "$$calibration_root" && test ! -L "$$calibration_root"; \
		calibration_digest="$$(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$(CURDIR):$(CURDIR)/src" /usr/bin/python3 -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1], environ={}).root_sha256)' "$$calibration_root")"; \
		test "$$calibration_digest" = "$(PHASE11_KVQUANT_CALIBRATION_ROOT_SHA256)"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		test -d "$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$phase12_artifact_root,dst=/home/rockrock/cmu_paper/artifacts/phase12,readonly" \
			--mount "type=bind,src=$$phase13b_bundle,dst=/home/rockrock/cmu_paper/artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission,readonly" \
			--mount "type=bind,src=$$phase13rq4_bundle,dst=/home/rockrock/cmu_paper/artifacts/phase13rq4/phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd,readonly" \
			--mount "type=bind,src=$$stage,dst=/home/rockrock/cmu_paper/$$stage_relative" \
			--mount "type=bind,src=$$task_root/kivi-source,dst=/opt/kivi-source,readonly" \
			--mount "type=bind,src=$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so,dst=/opt/kvbench/.phase3/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-source,dst=/opt/kvquant-source,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-build,dst=/opt/kvquant-build" \
			--mount "type=bind,src=$$prefix_campaign_root,dst=/opt/kvbench-prefix-states" \
			--mount "type=bind,src=$$calibration_root,dst=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3,readonly" \
			--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 --env PYTHONIOENCODING=utf-8 \
			--env PYTHONPATH=/opt/kivi-source:/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env LANG=C.UTF-8 --env TZ=UTC --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 --env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env TORCH_CUDA_ARCH_LIST=12.0+PTX \
			--env TRITON_CACHE_DIR=/root/.triton --env KVBENCH_KIVI_SOURCE_ROOT=/opt/kivi-source \
			--env KVBENCH_KVQUANT_SOURCE_ROOT=/opt/kvquant-source \
			--env KVBENCH_KVQUANT_CALIBRATION_ROOT=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3 \
			--env KVBENCH_KVQUANT_EXTENSION_SHA256="$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)" \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" --env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper --entrypoint /usr/bin/bash "$$image_id" \
			--noprofile --norc -eu -o pipefail -c \
			'cd /opt/kvquant-build && /opt/kvbench/.venv/bin/python setup_cuda.py build_ext --inplace && fresh_extension="$$(find /opt/kvquant-build -maxdepth 1 -type f -name "quant_cuda.*.so" -print -quit)" && test -n "$$fresh_extension" && /usr/bin/strip --strip-unneeded "$$fresh_extension" && test "$$(sha256sum "$$fresh_extension" | cut -d " " -f 1)" = "$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)" && export KVBENCH_KVQUANT_EXTENSION="$$fresh_extension" KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION="$$fresh_extension" && cd /home/rockrock/cmu_paper && /opt/kvbench/.venv/bin/python -m scripts.phase13_pilot --run-campaign --stage "/home/rockrock/cmu_paper/'"$$stage_relative"'" --campaign-id "'"$$campaign_id"'" --git-sha "'"$$head"'"')"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		analysis_cid="$$(docker create --read-only --network=none \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$stage,dst=/home/rockrock/cmu_paper/$$stage_relative" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--workdir /home/rockrock/cmu_paper --entrypoint /opt/kvbench/.venv/bin/python3 "$$analysis_image" \
			-m scripts.phase13_pilot --materialize-analysis --stage "/home/rockrock/cmu_paper/$$stage_relative")"; \
		[[ "$$analysis_cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$analysis_cid"; \
		docker rm -f "$$analysis_cid" >/dev/null; analysis_cid=""; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --finalize-staged-campaign --stage "$$stage" --campaign-id "$$campaign_id"; \
		final_root="$$repository_root/artifacts/phase13/$$campaign_id"; \
		test -d "$$final_root" && test ! -L "$$final_root"; \
		test ! -e "$$stage" && test ! -L "$$stage"; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --validate-campaign --artifact "$$final_root"; \
		printf 'PHASE13_CAMPAIGN_ID=%s\nPHASE13_CAMPAIGN_ARTIFACT=%s\n' "$$campaign_id" "$$final_root"; \
		preserve=0

validate-pilot:
	@test -n "$(PHASE13_CAMPAIGN_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE13_CAMPAIGN_ARTIFACT_required"}' >&2; exit 2; }
	@artifact="$$(realpath -e "$(PHASE13_CAMPAIGN_ARTIFACT)")"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test -d "$$artifact" && test ! -L "$(PHASE13_CAMPAIGN_ARTIFACT)"; \
		test "$$(dirname "$$artifact")" = "$$repository_root/artifacts/phase13"; \
		[[ "$$(basename "$$artifact")" =~ ^phase13-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}$$ ]]; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13_pilot --validate-campaign --artifact "$$artifact"

densify-pilot-knees: override MEASUREMENT_IMAGE_CONFIG_DIGEST := $(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)
densify-pilot-knees: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_phase13d_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"clean_committed_phase13d_tree_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase13d.XXXXXX)"; \
		cid=""; reference_cid=""; analysis_cid=""; stage=""; final_root=""; prefix_root=""; preserve=1; \
		cleanup() { \
			status=$$?; \
			if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$reference_cid" ]]; then docker rm -f "$$reference_cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$analysis_cid" ]]; then docker rm -f "$$analysis_cid" >/dev/null 2>&1 || true; fi; \
			if [[ -n "$$stage" && -d "$$stage" ]]; then printf '{"status":"PHASE13D_STAGING_EVIDENCE_PRESERVED","path":"%s"}\n' "$$stage" >&2; fi; \
			if [[ -n "$$prefix_root" && -d "$$prefix_root" ]]; then printf '{"status":"PHASE13D_PREFIX_STATES_PRESERVED","path":"%s"}\n' "$$prefix_root" >&2; fi; \
			if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"PHASE13D_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; \
			trap - EXIT; exit $$status; \
		}; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; repository_root="$$(git rev-parse --show-toplevel)"; \
		test "$$repository_root" = "$(CURDIR)"; \
		git cat-file -e "$$head:docs/plans/phase13d-knee-densification.md"; \
		git cat-file -e "$$head:docs/plans/phase13d-candidate-table.json"; \
		git cat-file -e "$$head:docs/plans/phase13d-execution-order.json"; \
		image_id="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)"; analysis_image="$(PHASE13_ANALYSIS_IMAGE_CONFIG_DIGEST)"; \
		test "$$(docker image inspect "$$image_id" --format '{{.Id}}')" = "$$image_id"; \
		test "$$(docker image inspect "$$analysis_image" --format '{{.Id}}')" = "$$analysis_image"; \
		source_campaign="$(PHASE13D_SOURCE_CAMPAIGN)"; \
		test -d "$$source_campaign" && test ! -L "$$source_campaign"; \
		test "$$($(PHASE13_HOST_PYTHON) -c 'import sys; from scripts.r2_artifact import validate_local_artifact; print(validate_local_artifact(sys.argv[1], environ={}).root_sha256)' "$$source_campaign")" = "$(PHASE13D_SOURCE_ROOT_SHA256)"; \
		campaign_id="$$($(PHASE13_HOST_PYTHON) -m scripts.phase13d_knee_densification --new-campaign-id --git-sha "$$head")"; \
		[[ "$$campaign_id" =~ ^phase13d-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}$$ ]]; \
		stage="$$($(PHASE13_HOST_PYTHON) -m scripts.phase13d_knee_densification --reserve-campaign --campaign-id "$$campaign_id" --git-sha "$$head")"; \
		case "$$stage" in "$$repository_root"/artifacts/phase13d/.kvbench-staging/"$$campaign_id".*.staging) ;; *) echo '{"status":"BLOCKED","reason":"unsafe_phase13d_stage"}' >&2; exit 2;; esac; \
		prefix_root="/home/rockrock/phase13d_prefix_states/$$campaign_id"; \
		test ! -e "$$prefix_root" && test ! -L "$$prefix_root"; \
		mkdir -p "$$(dirname "$$prefix_root")"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/repository" remote remove origin; \
		test "$$(git -C "$$task_root/repository" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env" && test ! -L "$$task_root/repository/.env"; \
		mkdir "$$task_root/repository/.venv" "$$task_root/repository/.phase3"; \
		ln -s /opt/kvbench/.venv/bin "$$task_root/repository/.venv/bin"; \
		ln -s /opt/kvbench/.venv/lib "$$task_root/repository/.venv/lib"; \
		ln -s /opt/kvbench/.venv/pyvenv.cfg "$$task_root/repository/.venv/pyvenv.cfg"; \
		ln -s /opt/kvbench/.phase3/site-packages "$$task_root/repository/.phase3/site-packages"; \
		stage_relative="$${stage#$$repository_root/}"; \
		mkdir -p "$$task_root/repository/$$(dirname "$$stage_relative")"; mkdir "$$task_root/repository/$$stage_relative"; \
		mkdir -p "$$task_root/repository/artifacts/phase12" "$$task_root/repository/artifacts/phase13" "$$task_root/repository/artifacts/phase13b" "$$task_root/repository/artifacts/phase13rq4" "$$task_root/repository/artifacts/phase13d"; \
		mkdir -p \
			"$$task_root/repository/artifacts/phase13/phase13-20260822t150835736582z-4ddd7b17-3a8fb3" \
			"$$task_root/repository/artifacts/phase13/phase13-20260804t111810342595z-a127b0d1-8649c3" \
			"$$task_root/repository/artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission" \
			"$$task_root/repository/artifacts/phase13rq4/phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd" \
			"$$task_root/repository/artifacts/phase13d/$$campaign_id"; \
		blocked_campaign="$$repository_root/artifacts/phase13/phase13-20260804t111810342595z-a127b0d1-8649c3"; \
		test -d "$$blocked_campaign" && test ! -L "$$blocked_campaign"; \
		docker run --rm --read-only --network=none --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$source_campaign,dst=/home/rockrock/cmu_paper/artifacts/phase13/phase13-20260822t150835736582z-4ddd7b17-3a8fb3,readonly" \
			--mount "type=bind,src=$$blocked_campaign,dst=/home/rockrock/cmu_paper/artifacts/phase13/phase13-20260804t111810342595z-a127b0d1-8649c3,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--workdir /home/rockrock/cmu_paper --entrypoint /usr/bin/bash "$$analysis_image" \
			--noprofile --norc -eu -o pipefail -c \
			'/opt/kvbench/.venv/bin/python3 -m scripts.phase13d_knee_densification --validate-source > /tmp/phase13d-source.json && /opt/kvbench/.venv/bin/python3 -m scripts.phase13d_knee_densification --validate-preregistration > /tmp/phase13d-preregistration.json'; \
		phase12_artifact_root="$$repository_root/artifacts/phase12"; \
		phase13b_bundle="$(PHASE13B_LOCAL_BUNDLE)"; phase13rq4_bundle="$(PHASE13RQ4_LOCAL_BUNDLE)"; \
		test "$$(realpath -e "$$phase13b_bundle")" = "$$phase13b_bundle"; \
		test "$$(realpath -e "$$phase13rq4_bundle")" = "$$phase13rq4_bundle"; \
		reference_image="$(KIVI_REFERENCE_IMAGE)@$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$reference_image" --format '{{.Id}}')" = "$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"; \
		test "$$(docker image inspect "$$reference_image" --format '{{index .Config.Labels "org.kvbench.reference.parent.config_digest"}}')" = "$$image_id"; \
		mkdir "$$task_root/kivi-source" "$$task_root/kivi-extension"; \
		reference_cid="$$(docker create --network=none "$$reference_image")"; \
		docker cp "$$reference_cid:/opt/kivi-source/." "$$task_root/kivi-source"; \
		docker cp "$$reference_cid:/opt/kivi-source/quant/kivi_gemv.cpython-312-x86_64-linux-gnu.so" "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so"; \
		docker rm -f "$$reference_cid" >/dev/null; reference_cid=""; \
		cp --preserve=mode,timestamps "$$task_root/kivi-source/models/kivi_gqa.py" "$$task_root/kivi-gqa.authority.py"; \
		git -C "$$task_root/kivi-source" clean -fdx; \
		cp --preserve=mode,timestamps "$$task_root/kivi-gqa.authority.py" "$$task_root/kivi-source/models/kivi_gqa.py"; \
		test "$$(sha256sum "$$task_root/kivi-source/quant/new_pack.py" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_NEW_PACK_SHA256)"; \
		test "$$(sha256sum "$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so" | cut -d ' ' -f 1)" = "$(PHASE8_KIVI_EXTENSION_SHA256)"; \
		$(PHASE13_HOST_PYTHON) scripts/validate_kivi_b019_patch.py --device cpu --source-root "$$task_root/kivi-source" > "$$task_root/kivi-source-validation.json"; \
		git clone --quiet --no-local --no-checkout "$(PHASE11DQ23_KVQUANT_SOURCE_ROOT)" "$$task_root/kvquant-source"; \
		git -C "$$task_root/kvquant-source" checkout --quiet --detach "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		git -C "$$task_root/kvquant-source" remote remove origin; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD)" = "$(PHASE11DQ23_KVQUANT_COMMIT)"; \
		test "$$(git -C "$$task_root/kvquant-source" rev-parse HEAD^{tree})" = "$(PHASE11DQ23_KVQUANT_TREE)"; \
		test -z "$$(git -C "$$task_root/kvquant-source" status --porcelain=v1 --untracked-files=all)"; \
		$(PHASE13_HOST_PYTHON) -m scripts.validate_kvquant_q23_long_context_patch --source-root "$$task_root/kvquant-source" > "$$task_root/kvquant-source-validation.json"; \
		mkdir "$$task_root/kvquant-build"; \
		for file in setup_cuda.py quant_cuda.cpp quant_cuda_kernel.cu measurement_cuda_kernel.cu; do relative="deployment/kvquant/$$file"; cp -- "$$task_root/kvquant-source/$$relative" "$$task_root/kvquant-build/$$file"; done; \
		calibration_root="$(PHASE11_KVQUANT_CALIBRATION)"; model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		test -d "$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		cid="$$(docker create --read-only --network=none --pid=host --gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,exec,nosuid,nodev,size=16g --tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$phase12_artifact_root,dst=/home/rockrock/cmu_paper/artifacts/phase12,readonly" \
			--mount "type=bind,src=$$phase13b_bundle,dst=/home/rockrock/cmu_paper/artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission,readonly" \
			--mount "type=bind,src=$$phase13rq4_bundle,dst=/home/rockrock/cmu_paper/artifacts/phase13rq4/phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd,readonly" \
			--mount "type=bind,src=$$stage,dst=/home/rockrock/cmu_paper/$$stage_relative" \
			--mount "type=bind,src=$$(dirname "$$prefix_root"),dst=/opt/kvbench-phase13d-prefix-states" \
			--mount "type=bind,src=$$task_root/kivi-source,dst=/opt/kivi-source,readonly" \
			--mount "type=bind,src=$$task_root/kivi-extension/kivi_gemv.cpython-312-x86_64-linux-gnu.so,dst=/opt/kvbench/.phase3/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-source,dst=/opt/kvquant-source,readonly" \
			--mount "type=bind,src=$$task_root/kvquant-build,dst=/opt/kvquant-build" \
			--mount "type=bind,src=$$calibration_root,dst=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3,readonly" \
			--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 --env PYTHONIOENCODING=utf-8 \
			--env PYTHONPATH=/opt/kivi-source:/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env LANG=C.UTF-8 --env TZ=UTC --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_DISABLE_TELEMETRY=1 --env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env TORCH_CUDA_ARCH_LIST=12.0+PTX --env TRITON_CACHE_DIR=/root/.triton \
			--env KVBENCH_KIVI_SOURCE_ROOT=/opt/kivi-source --env KVBENCH_KVQUANT_SOURCE_ROOT=/opt/kvquant-source \
			--env KVBENCH_KVQUANT_CALIBRATION_ROOT=/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3 \
			--env KVBENCH_KVQUANT_EXTENSION_SHA256="$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)" \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" --env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper --entrypoint /usr/bin/bash "$$image_id" --noprofile --norc -eu -o pipefail -c \
			'cd /opt/kvquant-build && /opt/kvbench/.venv/bin/python setup_cuda.py build_ext --inplace && fresh_extension="$$(find /opt/kvquant-build -maxdepth 1 -type f -name "quant_cuda.*.so" -print -quit)" && /usr/bin/strip --strip-unneeded "$$fresh_extension" && test "$$(sha256sum "$$fresh_extension" | cut -d " " -f 1)" = "$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)" && export KVBENCH_KVQUANT_EXTENSION="$$fresh_extension" KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION="$$fresh_extension" && cd /home/rockrock/cmu_paper && /opt/kvbench/.venv/bin/python -m scripts.phase13d_knee_densification --run-campaign --stage "/home/rockrock/cmu_paper/'"$$stage_relative"'" --campaign-id "'"$$campaign_id"'" --git-sha "'"$$head"'" --prefix-root "/opt/kvbench-phase13d-prefix-states/'"$$campaign_id"'"')"; \
		docker start --attach "$$cid"; docker rm -f "$$cid" >/dev/null; cid=""; \
		analysis_cid="$$(docker create --read-only --network=none --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$stage,dst=/home/rockrock/cmu_paper/$$stage_relative" \
			--mount "type=bind,src=$$source_campaign,dst=/home/rockrock/cmu_paper/artifacts/phase13/phase13-20260822t150835736582z-4ddd7b17-3a8fb3,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--workdir /home/rockrock/cmu_paper --entrypoint /opt/kvbench/.venv/bin/python3 "$$analysis_image" \
			-m scripts.phase13d_knee_densification --materialize-analysis --stage "/home/rockrock/cmu_paper/$$stage_relative")"; \
		docker start --attach "$$analysis_cid"; docker rm -f "$$analysis_cid" >/dev/null; analysis_cid=""; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13d_knee_densification --finalize-staged-campaign --stage "$$stage" --campaign-id "$$campaign_id"; \
		stage=""; final_root="$$repository_root/artifacts/phase13d/$$campaign_id"; \
		test -d "$$final_root" && test ! -L "$$final_root"; \
		analysis_cid="$$(docker create --read-only --network=none --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$task_root/repository,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$$source_campaign,dst=/home/rockrock/cmu_paper/artifacts/phase13/phase13-20260822t150835736582z-4ddd7b17-3a8fb3,readonly" \
			--mount "type=bind,src=$$final_root,dst=/home/rockrock/cmu_paper/artifacts/phase13d/$$campaign_id,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--workdir /home/rockrock/cmu_paper --entrypoint /opt/kvbench/.venv/bin/python3 "$$analysis_image" \
			-m scripts.phase13d_knee_densification --validate-campaign --artifact "/home/rockrock/cmu_paper/artifacts/phase13d/$$campaign_id")"; \
		docker start --attach "$$analysis_cid"; docker rm -f "$$analysis_cid" >/dev/null; analysis_cid=""; \
		printf 'PHASE13D_CAMPAIGN_ID=%s\nPHASE13D_CAMPAIGN_ARTIFACT=%s\nPHASE13D_PREFIX_ROOT=%s\n' "$$campaign_id" "$$final_root" "$$prefix_root"; \
		preserve=0

validate-pilot-densification:
	@test -n "$(PHASE13D_CAMPAIGN_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE13D_CAMPAIGN_ARTIFACT_required"}' >&2; exit 2; }
	@artifact="$$(realpath -e "$(PHASE13D_CAMPAIGN_ARTIFACT)")"; repository_root="$$(git rev-parse --show-toplevel)"; \
		test -d "$$artifact" && test ! -L "$(PHASE13D_CAMPAIGN_ARTIFACT)"; \
		test "$$(dirname "$$artifact")" = "$$repository_root/artifacts/phase13d"; \
		[[ "$$(basename "$$artifact")" =~ ^phase13d-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}$$ ]]; \
		analysis_image="$(PHASE13_ANALYSIS_IMAGE_CONFIG_DIGEST)"; \
		test "$$(docker image inspect "$$analysis_image" --format '{{.Id}}')" = "$$analysis_image"; \
		docker run --rm --read-only --network=none --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$repository_root,dst=/home/rockrock/cmu_paper,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--workdir /home/rockrock/cmu_paper --entrypoint /opt/kvbench/.venv/bin/python3 "$$analysis_image" \
			-m scripts.phase13d_knee_densification --validate-campaign --artifact "/home/rockrock/cmu_paper/artifacts/phase13d/$$(basename "$$artifact")"

test-phase13f:
	@$(MAKE) MEASUREMENT_IMAGE_CONFIG_DIGEST="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" verify-measurement-container
	@task_root="$$(mktemp -d /tmp/kvbench-phase13f-tests.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		head="$$(git rev-parse HEAD)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/repository" remote remove origin; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env" && test ! -L "$$task_root/repository/.env"; \
		mkdir -p "$$task_root/repository/artifacts/phase12" "$$task_root/repository/artifacts/phase13" "$$task_root/repository/artifacts/phase13b"; \
		docker run --rm --read-only --network=none \
			--tmpfs /tmp:rw,nosuid,nodev,size=512m \
			--mount "type=bind,src=$$task_root/repository,dst=/workspace,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase12,dst=/workspace/artifacts/phase12,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase13,dst=/workspace/artifacts/phase13,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase13b,dst=/workspace/artifacts/phase13b,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/workspace/src:/workspace \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /workspace --entrypoint /opt/kvbench/.venv/bin/python3 \
			"$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
			-m unittest tests.unit.test_phase13f_feasibility tests.unit.test_phase13_pilot -v

remediate-phase13-feasibility:
	@$(MAKE) MEASUREMENT_IMAGE_CONFIG_DIGEST="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" verify-measurement-container
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"clean_committed_phase13f_tree_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase13f.XXXXXX)"; \
		stage=""; preserve=1; \
		cleanup() { \
			status=$$?; \
			if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; \
			else printf '{"status":"PHASE13F_DIAGNOSTIC_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; \
			trap - EXIT; exit $$status; \
		}; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		identifier="phase13f-$$(date -u +%Y%m%dt%H%M%S%6Nz)-$${head:0:8}-$$(openssl rand -hex 3)"; \
		[[ "$$identifier" =~ ^phase13f-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}$$ ]]; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/repository"; \
		git -C "$$task_root/repository" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/repository" remote remove origin; \
		test -z "$$(git -C "$$task_root/repository" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/repository/.env" && test ! -L "$$task_root/repository/.env"; \
		mkdir -p "$$task_root/repository/artifacts/phase12" "$$task_root/repository/artifacts/phase13" "$$task_root/repository/artifacts/phase13b"; \
		mkdir "$$task_root/output"; stage="$$task_root/output/$$identifier"; mkdir "$$stage"; \
		docker run --rm --read-only --network=none \
			--tmpfs /tmp:rw,nosuid,nodev,size=512m \
			--mount "type=bind,src=$$task_root/repository,dst=/workspace,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase12,dst=/workspace/artifacts/phase12,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase13,dst=/workspace/artifacts/phase13,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase13b,dst=/workspace/artifacts/phase13b,readonly" \
			--mount "type=bind,src=$$task_root/output,dst=/evidence" \
			--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/workspace/src:/workspace \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /workspace --entrypoint /opt/kvbench/.venv/bin/python3 \
			"$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
			-m scripts.phase13f_feasibility generate --output "/evidence/$$identifier" --git-sha "$$head"; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13f_feasibility validate "$$stage"; \
		final="$$( $(PHASE13_HOST_PYTHON) -m scripts.phase13f_feasibility promote "$$stage" | $(PHASE2_PYTHON) -c 'import json,sys; print(json.load(sys.stdin)["artifact_path"])' )"; \
		test -d "$$final" && test ! -L "$$final"; \
		preserve=0; \
		printf 'PHASE13F_ARTIFACT=%s\n' "$$final"

validate-phase13-feasibility:
	@test -n "$(PHASE13F_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE13F_ARTIFACT_required"}' >&2; exit 2; }
	@artifact="$$(realpath -e "$(PHASE13F_ARTIFACT)")"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test -d "$$artifact" && test ! -L "$(PHASE13F_ARTIFACT)"; \
		test "$$(dirname "$$artifact")" = "$$repository_root/artifacts/phase13f"; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13f_feasibility validate "$$artifact"

test-phase13t:
	@$(MAKE) MEASUREMENT_IMAGE_CONFIG_DIGEST="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" verify-measurement-container
	@docker run --rm --read-only --network=none \
		--tmpfs /tmp:rw,nosuid,nodev,size=512m \
		--mount "type=bind,src=$(CURDIR),dst=/workspace,readonly" \
		--mount "type=bind,src=$(CURDIR)/artifacts/phase13,dst=/workspace/artifacts/phase13,readonly" \
		--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 \
		--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/workspace/src:/workspace \
		--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
		--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
		--workdir /workspace --entrypoint /opt/kvbench/.venv/bin/python3 \
		"$(PHASE13_AUTHORIZED_IMAGE_CONFIG_DIGEST)" \
		-m unittest tests.unit.test_phase13t_timeout tests.unit.test_phase8_process_supervision -v

validate-phase13-timeout:
	@test -n "$(PHASE13T_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE13T_ARTIFACT_required"}' >&2; exit 2; }
	@artifact="$$(realpath -e "$(PHASE13T_ARTIFACT)")"; \
		repository_root="$$(git rev-parse --show-toplevel)"; \
		test -d "$$artifact" && test ! -L "$(PHASE13T_ARTIFACT)"; \
		test "$$(dirname "$$artifact")" = "$$repository_root/artifacts/phase13t"; \
		$(PHASE13_HOST_PYTHON) -m scripts.phase13t_timeout validate "$$artifact"

full-scan:
	@$(PHASE2_CLI) run --plan configs/plans/full_scan.yaml --dry-run
	@echo '{"status":"validation_only","target":"full-scan","timing_collected":false}'

test-phase16:
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_phase16_full_scan tests.unit.test_phase16_logical_prefix -v

validate-full-scan:
	@test -n "$(PHASE16_FULL_SCAN_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE16_FULL_SCAN_ARTIFACT_required"}' >&2; exit 2; }
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase16_full_scan --validate-full-scan --artifact "$(PHASE16_FULL_SCAN_ARTIFACT)"

profile-subset:
	@$(PHASE2_CLI) run --plan configs/plans/profiler_subset.yaml --dry-run
	@echo '{"status":"validation_only","target":"profile-subset","profiler_executed":false}'

PHASE17_PYTHON ?= .phase17-venv/bin/python
PHASE17_ARTIFACT ?=

test-phase17:
	@PYTHONPATH=src $(PHASE17_PYTHON) -m unittest tests.unit.test_phase17_modeling -v

fit:
	@PYTHONPATH=src $(PHASE17_PYTHON) -m scripts.phase17_modeling

validate-phase17-modeling:
	@test -n "$(PHASE17_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE17_ARTIFACT_required"}' >&2; exit 2; }
	@PYTHONPATH=src $(PHASE17_PYTHON) -m scripts.phase17_modeling --validate "$(PHASE17_ARTIFACT)"

figures:
	@echo '{"error":"phase_not_implemented","target":"figures","phase":"17+"}' >&2
	@exit 2

PHASE18_SOURCE_PACKAGE ?= artifacts/phase18/phase18-current
PHASE18_OUTPUT ?= /tmp/kvbench-phase18-reproduced
PHASE18_ARTIFACT ?=

test-phase18:
	@PYTHONPATH=src $(PHASE17_PYTHON) -m unittest tests.unit.test_phase18_reproduction -v

reproduce:
	@python3 "$(PHASE18_SOURCE_PACKAGE)/reproduce.py" reproduce --package "$(PHASE18_SOURCE_PACKAGE)" --output "$(PHASE18_OUTPUT)"

package-phase18:
	@PYTHONPATH=src $(PHASE17_PYTHON) -m scripts.phase18_package

validate-phase18:
	@test -n "$(PHASE18_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"PHASE18_ARTIFACT_required"}' >&2; exit 2; }
	@PYTHONPATH=src $(PHASE17_PYTHON) -m scripts.phase18_package --validate "$(PHASE18_ARTIFACT)"
