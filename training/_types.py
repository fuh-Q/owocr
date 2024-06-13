"""
Typeshed for certain Keras and TensorFlow types that are incomplete

These types aren't fully correct and verified but it suffices for what I need
"""

from __future__ import annotations

from typing import Any, List, Mapping, Protocol, TypedDict, Tuple, TYPE_CHECKING
from typing_extensions import Self  # type: ignore
if TYPE_CHECKING:
    import tensorflow as tf
    from tensorflow.keras import Variable
    from tensorflow.keras.initializers import Initializer  # type: ignore
    from tensorflow.keras.optimizers.schedules import LearningRateSchedule  # type: ignore

    Var = tf.Variable | Variable

__all__ = (
    "HeNormalLike",
    "SGDLike"
)

class HeNormalCfg(TypedDict):
    seed: int

class SGDCfg(TypedDict):
    name: str
    learning_rate: float | LearningRateSchedule
    weight_decay: float | None
    clipnorm: float | None
    global_clipnorm: float | None
    clipvalue: float | None
    use_ema: bool
    ema_momentum: float
    ema_overwrite_frequency: int | None
    loss_scale_factor: float | None
    gradient_accumulation_steps: int | None
    momentum: float
    nesterov: bool

class HeNormalLike(Protocol):
    distribution: str
    mode: str
    scale: float
    seed: int
    @classmethod
    def from_config(cls, config: HeNormalCfg) -> Self: ...
    def get_config(self) -> HeNormalCfg: ...
    def clone(self) -> Self: ...

class SGDLike(Protocol):
    built: bool
    clipnorm: float | None
    clipvalue: float | None
    ema_momentum: float
    ema_overwrite_frequency: int | None
    global_clipnorm: float | None
    gradient_accumulation_steps: int | None
    iterations: Variable
    learning_rate: float | LearningRateSchedule
    loss_scale_factor: float | None
    momentum: float
    name: str
    nesterov: bool
    use_ema: bool
    variables: List[Variable]
    weight_decay: float | None
    def add_variable(
        self,
        shape: tf.TensorShape,
        initializer: str | Initializer,
        dtype: str | tf.DType,
        aggregation: tf.VariableAggregation,
        name: str = ...
    ) -> Var: ...
    def add_variable_from_reference(
        self,
        reference_variable: Var,
        initializer: str | Initializer,
        name: str = ...
    ) -> Var: ...
    def apply(self, grads: List[tf.Tensor], trainable_variables: List[Var] | None): ...
    def apply_gradients(self, grads_and_vars: List[List[tf.Tensor] | List[Var]]): ...
    def assign(self, variable: Var, value: Any): ...
    def assign_add(self, variable: Var, value: Any): ...
    def assign_sub(self, variable: Var, value: Any): ...
    def build(self, var_list: List[Var]): ...
    def exclude_from_weight_decay(self, var_list: List[Var], var_names: List[str]): ...
    def finalize_variable_values(self, var_list: List[Var]): ...
    @classmethod
    def from_config(cls, config: SGDCfg) -> Self: ...
    def get_config(self) -> SGDCfg: ...
    def load_own_variables(self, store: Mapping[str, Var]): ...
    def save_own_variables(self, store: Mapping[str, Var]): ...
    def scale_loss(self, loss: float) -> float: ...
    def set_weights(self, weights: List[Any]): ...
    def stateless_apply(
        self,
        optimizer_variables: List[Var],
        grads: List[tf.Tensor],
        trainable_variables: List[Var]
    ) -> Tuple[List[Var], List[Var]]: ...
    def update_step(self, gradient: tf.Tensor, variable: Var, learning_rate: float | LearningRateSchedule): ...
