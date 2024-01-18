# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.manager import ExportContext

from .handler import TorchCDQCastBiasQuantProxyHandler
from .handler import TorchCDQCastDecoupledWeightQuantProxyHandler
from .handler import TorchCDQCastDecoupledWeightQuantWithInputProxyHandler
from .handler import TorchCDQCastWeightQuantProxyHandler
from .handler import TorchQCDQCastActQuantProxyHandler
from .handler import TorchQCDQCastTruncQuantProxyHandler


class TorchQCDQManager(BaseManager):
    target_name = 'torch'

    handlers = [
        TorchCDQCastWeightQuantProxyHandler,
        TorchCDQCastDecoupledWeightQuantProxyHandler,
        TorchCDQCastDecoupledWeightQuantWithInputProxyHandler,
        TorchQCDQCastActQuantProxyHandler,
        TorchCDQCastBiasQuantProxyHandler,
        TorchQCDQCastTruncQuantProxyHandler]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)

    @classmethod
    def export(cls, module: Module, args, export_path: Optional[str] = None):
        with ExportContext(cls):
            traced_module = cls.jit_inference_trace(module, args, export_path)
        return traced_module
