from typing import Optional, Any
import jax.numpy as jnp

class ConvergenceTracker:
    def __init__(
        self, 
        convergence: list[float], 
        learning_rate: float, 
        ema_alpha: float, 
        min_steps_per_threshold: int
    ):
        self.ema_alpha = ema_alpha
        self.min_steps_per_threshold = min_steps_per_threshold
        
        # Sort thresholds descending and scale by learning rate
        self.convergence_thresholds = [ct * learning_rate for ct in sorted(convergence, reverse=True)]
        self.current_threshold_idx = 0
        self.current_threshold = self.convergence_thresholds[self.current_threshold_idx]
        
        self.ema_loss_delta: Optional[Any] = None
        self.ema_params: Optional[Any] = None
        self.steps_since_threshold_start = 0

    def update(self, previous_loss: Optional[Any], current_loss: Any, current_params: Any) -> Any:
        """
        Calculates raw loss delta and updates EMA stats if a previous loss exists.
        Returns the raw loss delta.
        """
        if previous_loss is not None:
            raw_loss_delta = jnp.abs(previous_loss - current_loss)
            
            if self.ema_loss_delta is None or self.ema_params is None:
                self.ema_loss_delta = raw_loss_delta
                self.ema_params = current_params
            else:
                self.ema_loss_delta = self.ema_alpha * raw_loss_delta + (1 - self.ema_alpha) * self.ema_loss_delta
                self.ema_params = self.ema_alpha * current_params + (1 - self.ema_alpha) * self.ema_params
        else:
            raw_loss_delta = 0.0
            
        self.steps_since_threshold_start += 1
        return raw_loss_delta
        
    def get_relative_convergence(self, current_loss: Any) -> float:
        if self.ema_loss_delta is not None and current_loss > 0:
            return float(self.ema_loss_delta / current_loss)
        return 0.0

    def is_threshold_met(self, current_loss: Any, step: int, initial_steps: int) -> bool:
        if (
            self.steps_since_threshold_start >= self.min_steps_per_threshold
            and self.ema_loss_delta is not None
            and self.get_relative_convergence(current_loss) < self.current_threshold
            and step > initial_steps
        ):
            return True
        return False
        
    def advance_threshold(self) -> bool:
        """Advances to the next threshold. Returns False if all thresholds are completed."""
        self.current_threshold_idx += 1
        self.steps_since_threshold_start = 0
        
        if self.current_threshold_idx >= len(self.convergence_thresholds):
            return False
            
        self.current_threshold = self.convergence_thresholds[self.current_threshold_idx]
        return True
        
    def reset_threshold_steps(self) -> None:
        self.steps_since_threshold_start = 0
