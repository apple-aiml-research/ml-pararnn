#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as typ
import typing_inspect
import dataclasses
from dataclasses import dataclass, field

import torch

from pararnn.parallel_reduction.parallel_reduction import NewtonConfig

TraitT = typ.TypeVar("TraitT")
T = typ.TypeVar("T")


class TraitCheckMixin:
    @classmethod
    @typ.final
    def trait(cls) -> typ.Type[TraitT]:
        generic_base_type, = typing_inspect.get_generic_bases(cls)
        trait, = typing_inspect.get_args(generic_base_type)
        return trait


@dataclass
class Config(TraitCheckMixin, typ.Generic[TraitT]):
    """
    Common config to all Recurrent Models
    """
    state_dim: torch.int
    input_dim: torch.int
    device: torch.device = torch.device('cpu')
    dtype: torch.dtype = torch.float32
    mode: str = 'parallel'
    newton_config: NewtonConfig = field(default_factory=NewtonConfig)


@dataclass
class SystemParameters(TraitCheckMixin, typ.Generic[TraitT]):
    """
    Placeholder for Parameters defining a Recurrent Model

    Note: I need to make this iterable for compatibility with torch.autograd.function.forward
    """
    
    # TODO: ditch dataclass and go for some custom iterable class instead?
    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))
    
    @classmethod
    def repack(
            cls: T,
            pars: typ.Tuple[typ.Union[torch.Tensor, typ.Callable, None], ...]
    ) -> T:
        raise NotImplementedError()

    @classmethod
    def unpack(cls: T) -> typ.Tuple[typ.Union[torch.Tensor, typ.Callable, None]]:
        raise NotImplementedError


# TODO: maybe better define these inside each class that employs them, rather than here once and for all?
ConfigT = typ.TypeVar("ConfigT", bound=Config)
SystemParametersT = typ.TypeVar("SystemParametersT", bound=SystemParameters)

