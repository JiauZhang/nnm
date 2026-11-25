import glob
import importlib.util
import json
import os
from pathlib import Path

import torch
from conippets.config import Config
from torch import nn

MANIFEST_NAME = "nnm_weights.json"
DEFAULT_PATTERNS = ("model.safetensors", "model-*.safetensors", "*.safetensors")
PROVIDERS = ("huggingface", "modelscope")
PROVIDER_ALIASES = {"hf": "huggingface", "ms": "modelscope"}


def download_kwargs(source, provider, repo, pattern=None, revision=None):
    if provider == "huggingface":
        kwargs = {"repo_id": repo}
        names = (
            "cache_dir",
            "revision",
            "local_files_only",
            "token",
            "force_download",
            "resume_download",
            "endpoint",
            "proxies",
        )
    else:
        kwargs = {"model_id": repo}
        names = ("cache_dir", "revision")

    for name in names:
        value = revision if name == "revision" and revision is not None else source.get(name)
        if value is not None:
            kwargs[name] = value
    if pattern is not None:
        kwargs["allow_patterns"] = [pattern]
    return kwargs


def resolve_device(preferred):
    if preferred is None:
        preferred = "auto"
    if isinstance(preferred, torch.device):
        return preferred
    value = str(preferred).lower()
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_dtype(tensor, dtype):
    if dtype is None or not tensor.is_floating_point():
        return tensor
    return tensor.to(dtype=dtype)


class PretrainedModel(nn.Module):
    weight_pattern = None
    module_name = None

    @classmethod
    def from_pretrained(cls, pretrained_path, *, source=None, weights=None, config=None):
        source = Config.from_dict(source or {})
        weights = Config.from_dict(weights or {})
        overrides = Config.from_dict(config or {})

        device = resolve_device(weights.get("device", "auto"))
        dtype = weights.get("dtype")
        strict = weights.get("strict", True)
        low_cpu_mem_usage = weights.get("low_cpu_mem_usage", False)

        directory = cls._resolve_pretrained_path(pretrained_path, source)
        manifest = cls._load_manifest(directory)
        entry = cls._resolve_entry(manifest, overrides)
        config = cls.load_config(directory, entry)
        config.update(overrides)

        if low_cpu_mem_usage:
            with torch.device("meta"):
                model = cls(config)
        else:
            model = cls(config)
            if dtype is not None:
                model.to(dtype=dtype)
            model.to(device)

        state_dict = cls._load_state_dict(directory, config, entry, source, weights, device)
        if low_cpu_mem_usage:
            model.load_state_dict(state_dict, strict=strict, assign=True)
        else:
            model.load_state_dict(state_dict, strict=strict)
        model._post_load(directory)
        return model

    def _post_load(self, directory):
        pass

    @classmethod
    def load_config(cls, directory, manifest=None):
        relative = manifest.get("config", "config.json") if manifest else "config.json"
        config_path = os.path.join(directory, relative)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return Config.from_json(config_path)

    @classmethod
    def _load_manifest(cls, directory):
        path = os.path.join(directory, MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        manifest = json.loads(Path(path).read_text())
        if manifest.get("version") != 1:
            raise ValueError(f"Unsupported {MANIFEST_NAME} version: {manifest.get('version')!r}")
        return manifest

    @classmethod
    def manifest_key(cls, config):
        return cls.module_name

    @classmethod
    def _resolve_entry(cls, manifest, config):
        if manifest is None or cls.module_name is None:
            return manifest
        modules = manifest.get("modules", {})
        key = cls.manifest_key(config)
        if key not in modules:
            raise KeyError(f"{MANIFEST_NAME} has no module {key!r}")
        return {**manifest, **modules[key]}

    @classmethod
    def _resolve_pretrained_path(cls, pretrained_path, source):
        path = os.path.expanduser(str(pretrained_path))
        if os.path.exists(path):
            return path
        if os.path.isabs(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        provider, repo, subdirectory = cls._split_repo(pretrained_path)
        snapshot = cls._download(repo, provider, source)
        return os.path.join(snapshot, subdirectory) if subdirectory else snapshot

    @classmethod
    def _split_repo(cls, repo_id):
        value = str(repo_id)
        provider = "huggingface"
        if ":" in value:
            prefix, value = value.split(":", 1)
            provider = PROVIDER_ALIASES.get(prefix, prefix)
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider {provider!r}; expected one of {PROVIDERS}")

        parts = value.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid repo id {repo_id!r}; expected owner/name")
        repo = "/".join(parts[:2])
        subdirectory = os.path.join(*parts[2:]) if len(parts) > 2 else None
        return provider, repo, subdirectory

    @classmethod
    def _download(cls, repo, provider, source, pattern=None, revision=None):
        kwargs = download_kwargs(source, provider, repo, pattern, revision)
        if provider == "huggingface":
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        else:
            from modelscope import snapshot_download  # noqa: PLC0415
        return snapshot_download(**kwargs)

    @classmethod
    def _load_state_dict(cls, directory, config, manifest, source, weights, device):
        paths, mapping = cls._resolve_weight_files(directory, config, manifest, source)
        if not paths:
            raise FileNotFoundError(f"No weight files found for {directory}")
        if weights.get("low_cpu_mem_usage", False):
            return cls._load_state_dict_low_cpu(directory, paths, mapping, config, weights, device)

        state_dict = {}
        for path in paths:
            file_state = cls._read_file(path)
            if file_state:
                state_dict.update(file_state)
        if mapping:
            state_dict = cls._apply_mapping(state_dict, directory, mapping, config)
        return state_dict

    @classmethod
    def _resolve_weight_files(cls, directory, config, manifest, source):
        paths = []
        mapping = None
        if manifest:
            for item in manifest.get("sources", []):
                provider = item.get("provider", "huggingface")
                pattern = item.get("pattern")
                snapshot = cls._download(item["repo"], provider, source, pattern=pattern, revision=item.get("revision"))
                paths.extend(cls._matching_paths(snapshot, pattern))
            paths.extend(os.path.join(directory, name) for name in manifest.get("weights", []))
            mapping = manifest.get("mapping")
        else:
            pattern = getattr(config, "weight_pattern", cls.weight_pattern)
            paths = cls._matching_paths(directory, pattern) if pattern else cls._auto_paths(directory)
            if os.path.exists(os.path.join(directory, "weight_mapping.py")):
                mapping = "weight_mapping.py"
        return paths, mapping

    @classmethod
    def _index_paths(cls, directory):
        path = os.path.join(directory, "model.safetensors.index.json")
        if not os.path.exists(path):
            return []
        index = json.loads(Path(path).read_text())
        return sorted({os.path.join(directory, name) for name in index.get("weight_map", {}).values()})

    @classmethod
    def _matching_paths(cls, directory, pattern):
        indexed = cls._index_paths(directory)
        if indexed:
            return indexed
        if not pattern:
            return cls._auto_paths(directory)
        return sorted(glob.glob(os.path.join(directory, pattern)))

    @classmethod
    def _auto_paths(cls, directory):
        indexed = cls._index_paths(directory)
        if indexed:
            return indexed
        for pattern in DEFAULT_PATTERNS:
            paths = sorted(glob.glob(os.path.join(directory, pattern)))
            if paths:
                return paths
        return []

    @classmethod
    def _iter_tensor_items(cls, paths):
        for path in paths:
            if path.endswith(".safetensors"):
                from safetensors.torch import safe_open  # noqa: PLC0415

                with safe_open(path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        yield key, handle.get_tensor(key)
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                continue
            state = checkpoint.get("state_dict", checkpoint)
            for key, value in state.items():
                yield key, value

    @classmethod
    def _load_mapping_module(cls, directory, mapping):
        path = os.path.join(directory, mapping)
        spec = importlib.util.spec_from_file_location("nnm_weight_mapping", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _map_key_func(cls, module, config):
        if hasattr(module, "map_key"):
            return lambda key: module.map_key(key, config)

        def map_key(key):
            mapped = module.map_state_dict({key: torch.empty(0)}, config)
            if len(mapped) != 1:
                raise ValueError("Mapping changes tensor cardinality; low_cpu_mem_usage requires map_key")
            return next(iter(mapped))

        return map_key

    @classmethod
    def _load_state_dict_low_cpu(cls, directory, paths, mapping, config, weights, device):
        module = cls._load_mapping_module(directory, mapping) if mapping else None
        map_key = cls._map_key_func(module, config) if module else lambda key: key
        state_dict = {}
        for key, value in cls._iter_tensor_items(paths):
            target = map_key(key)
            if target in state_dict:
                raise ValueError(f"Duplicate mapped key: {target}")
            value = apply_dtype(value, weights.get("dtype"))
            if device.type != "cpu":
                value = value.to(device)
            state_dict[target] = value
        return state_dict

    @classmethod
    def _read_file(cls, path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file  # noqa: PLC0415

            return load_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return None
        return checkpoint.get("state_dict", checkpoint)

    @classmethod
    def _apply_mapping(cls, state_dict, directory, mapping, config):
        module = cls._load_mapping_module(directory, mapping)
        return module.map_state_dict(state_dict, config)
