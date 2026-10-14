# Copyright (c) ModelScope Contributors. All rights reserved.
"""Biren SUPA platform support."""

from .base import Platform


def ensure_supa_backend() -> None:
    """Load SUPA and install its CUDA compatibility surface."""
    try:
        import torch_supa  # noqa: F401
        # Importing this module installs the compatibility patch; this release
        # does not expose a callable transfer_to_supa symbol.
        import torch_supa.contrib.transfer_to_supa  # noqa: F401
    except Exception as exc:
        raise RuntimeError('SUPA backend is not available. Please install torch_supa.') from exc


class SUPA(Platform):
    @staticmethod
    def visible_device_env():
        return 'SUPA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'supa'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'supa:{idx}'

    @staticmethod
    def device_backend(platform: str = None):
        return 'bccl'

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0) -> str:
        raise NotImplementedError('vLLM SUPA device UUID is not implemented')

    @staticmethod
    def get_device_rng_state():
        import torch
        return torch.cuda.get_rng_state()

    @staticmethod
    def set_device_rng_state(state) -> None:
        import torch
        torch.cuda.set_rng_state(state)
