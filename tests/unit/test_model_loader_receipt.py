from __future__ import annotations

import dataclasses
from types import SimpleNamespace
import unittest
from unittest import mock

import kvbench.runtime.model_loader as model_loader


class FakeStorage:
    def __init__(
        self,
        *,
        pointer: int = 10_000,
        handle: int = 20_000,
        nbytes: int = 16,
    ) -> None:
        self.pointer = pointer
        self._cdata = handle
        self.byte_count = nbytes

    def data_ptr(self) -> int:
        return self.pointer

    def nbytes(self) -> int:
        return self.byte_count


class FakeParameter:
    def __init__(
        self,
        *,
        pointer: int = 10_000,
        handle: int = 20_000,
    ) -> None:
        self.shape = (8,)
        self.dtype = "torch.bfloat16"
        self.device = "cuda:0"
        self.requires_grad = False
        self._version = 0
        self.storage = FakeStorage(pointer=pointer, handle=handle)
        self.payload_marker = "trusted-load"

    def stride(self) -> tuple[int]:
        return (1,)

    def numel(self) -> int:
        return 8

    def data_ptr(self) -> int:
        return self.storage.pointer

    def untyped_storage(self) -> FakeStorage:
        return self.storage

    def storage_offset(self) -> int:
        return 0

    def element_size(self) -> int:
        return 2


class FakeModel:
    def __init__(self, parameter: FakeParameter | None = None) -> None:
        self.parameter = parameter or FakeParameter()
        self.training = True
        self.config = SimpleNamespace(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            max_position_embeddings=131_072,
            _attn_implementation=model_loader.ATTENTION_IMPLEMENTATION,
        )

    def to(self, *, device: object, dtype: object) -> FakeModel:
        del device, dtype
        return self

    def eval(self) -> FakeModel:
        self.training = False
        return self

    def requires_grad_(self, value: bool) -> FakeModel:
        self.parameter.requires_grad = value
        return self

    def parameters(self) -> tuple[FakeParameter, ...]:
        return (self.parameter,)

    def named_parameters(self) -> tuple[tuple[str, FakeParameter], ...]:
        return (("weight", self.parameter),)


FakeModel.__name__ = "LlamaForCausalLM"


class FakeBackendTokenizer:
    def __init__(self) -> None:
        self.serialized = (
            '{"normalizer":{"type":"Sequence"},'
            '"pre_tokenizer":{"type":"ByteLevel"},'
            '"post_processor":{"type":"TemplateProcessing"}}'
        )

    def to_str(self) -> str:
        return self.serialized


class FakeTokenizer:
    def __init__(self) -> None:
        self.backend_tokenizer = FakeBackendTokenizer()
        self.special_tokens_map_extended: dict[str, object] = {}
        self.added_tokens_decoder: dict[int, object] = {}
        self.all_special_ids = (128_000, 128_001)
        self.chat_template = "{{ messages }}"
        self.model_max_length = 131_072
        self.padding_side = "right"
        self.truncation_side = "right"

    def __len__(self) -> int:
        return 128_256


FakeTokenizer.__name__ = "PreTrainedTokenizerFast"


def frozen_identity() -> model_loader.FrozenModelIdentity:
    return model_loader.FrozenModelIdentity(
        model_id=model_loader.MODEL_ID,
        revision=model_loader.MODEL_REVISION,
        snapshot_path="/verified/snapshot",
        file_hashes={"weight.safetensors": "a" * 64},
        architecture="LlamaForCausalLM",
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=131_072,
        weight_dtype="bfloat16",
    )


def load_fake(
    *,
    model: FakeModel | None = None,
    tokenizer: FakeTokenizer | None = None,
) -> model_loader.LoadedFrozenModel:
    selected_model = model or FakeModel()
    selected_tokenizer = tokenizer or FakeTokenizer()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            del args, kwargs
            return selected_tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            del args, kwargs
            return selected_model

    fake_torch = SimpleNamespace(
        bfloat16="torch.bfloat16",
        device=lambda value: value,
    )
    fake_transformers = SimpleNamespace(
        __version__=model_loader.TRANSFORMERS_VERSION,
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
    )

    def import_module(name: str) -> object:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        raise ModuleNotFoundError(name)

    with (
        mock.patch.object(
            model_loader,
            "verify_frozen_snapshot",
            return_value=frozen_identity(),
        ),
        mock.patch.object(model_loader.importlib, "import_module", import_module),
        mock.patch.object(model_loader, "register_transformers_attention"),
    ):
        return model_loader.load_frozen_model(
            snapshot_path="/verified/snapshot",
            device="cuda:0",
        )


class FrozenModelReceiptTests(unittest.TestCase):
    def test_trusted_loader_constructs_and_revalidates_receipt(self) -> None:
        loaded = load_fake()
        model_loader.validate_loaded_frozen_model_receipt(loaded)
        self.assertIs(loaded.receipt._model_ref, loaded.model)
        self.assertIs(loaded.receipt._tokenizer_ref, loaded.tokenizer)
        self.assertIs(
            loaded.receipt._parameter_refs[0],
            loaded.model.parameter,
        )
        self.assertIs(
            loaded.receipt._storage_refs[0],
            loaded.model.parameter.storage,
        )

    def test_no_arbitrary_object_receipt_factory_exists(self) -> None:
        self.assertFalse(
            hasattr(model_loader, "_create_loaded_frozen_model")
        )

    def test_wrong_receipt_and_loaded_model_seals_fail(self) -> None:
        loaded = load_fake()
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "not factory sealed",
        ):
            dataclasses.replace(loaded.receipt, _seal=object())
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "not factory sealed",
        ):
            dataclasses.replace(loaded, _seal=object())

    def test_same_metadata_parameter_replacement_fails(self) -> None:
        loaded = load_fake()
        replacement = FakeParameter(pointer=10_000, handle=20_000)
        loaded.model.parameter = replacement
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "strong object or storage identity changed",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_storage_replacement_same_identity_numbers_fails(self) -> None:
        loaded = load_fake()
        loaded.model.parameter.storage = FakeStorage(
            pointer=10_000,
            handle=20_000,
        )
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "strong object or storage identity changed",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_parameter_version_increment_fails(self) -> None:
        loaded = load_fake()
        loaded.model.parameter._version += 1
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "no longer matches live objects",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_model_substitution_fails_against_strong_reference(self) -> None:
        loaded = load_fake()
        object.__setattr__(loaded, "model", FakeModel())
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "strong object or storage identity changed",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_tokenizer_substitution_fails_against_strong_reference(self) -> None:
        loaded = load_fake()
        object.__setattr__(loaded, "tokenizer", FakeTokenizer())
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "strong object or storage identity changed",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_tokenizer_backend_behavior_mutation_fails(self) -> None:
        loaded = load_fake()
        loaded.tokenizer.backend_tokenizer.serialized = (
            '{"normalizer":{"type":"Replace"}}'
        )
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "no longer matches live objects",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_tokenizer_chat_template_mutation_fails(self) -> None:
        loaded = load_fake()
        loaded.tokenizer.chat_template = "mutated"
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "no longer matches live objects",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_loader_source_change_after_import_invalidates_receipt(self) -> None:
        loaded = load_fake()
        with mock.patch.object(
            model_loader.Path,
            "read_bytes",
            return_value=b"mutated-loader-source",
        ):
            with self.assertRaisesRegex(
                model_loader.ModelIdentityError,
                "source changed after module import",
            ):
                model_loader.validate_loaded_frozen_model_receipt(loaded)

    def test_receipt_contract_does_not_claim_live_parameter_content_hash(
        self,
    ) -> None:
        loaded = load_fake()
        payload = loaded.receipt._payload()
        self.assertEqual(
            payload["parameter_binding_kind"],
            model_loader.PARAMETER_BINDING_KIND,
        )
        self.assertNotIn("parameter_content_sha256", payload)
        self.assertIn("no_live_content_hash", model_loader.PARAMETER_BINDING_KIND)

    def test_snapshot_file_ledger_is_bound(self) -> None:
        loaded = load_fake()
        loaded.identity.file_hashes["weight.safetensors"] = "b" * 64
        with self.assertRaisesRegex(
            model_loader.ModelIdentityError,
            "no longer matches live objects",
        ):
            model_loader.validate_loaded_frozen_model_receipt(loaded)


if __name__ == "__main__":
    unittest.main()
