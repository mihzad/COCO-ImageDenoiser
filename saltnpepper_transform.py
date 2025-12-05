from torchvision.transforms import v2
import torch
import torch.nn as nn

class BatchSaltAndPepper(v2.Transform):
    """
    Applies Salt and Pepper noise directly to a batch of PyTorch tensors (B, C, H, W).
    Designed for high performance and TPU compatibility.
    """
    def __init__(self, salt_prob: float = 0.01, pepper_prob: float = 0.01):
        super().__init__()
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.total_prob = salt_prob + pepper_prob

        if not (0.0 <= self.total_prob <= 1.0):
            raise ValueError("salt_prob + pepper_prob must be between 0.0 and 1.0")

    def _apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        # NOTE: This assumes the batch values are scaled between 0.0 and 1.0

        B, C, H, W = batch.shape

        # 1. Generate a random tensor on the same device (CPU/GPU/XLA) as the input batch
        # This is CRITICAL for TPU compatibility.
        rand_tensor = torch.rand(
            B, 1, H, W,
            device=batch.device,
            dtype=batch.dtype
        )

        # 2. Determine which pixels are Salt and which are Pepper using vectorized logic

        # Pixels where rand_tensor < salt_prob are 'Salt' (set to 1.0)
        salt_mask = rand_tensor < self.salt_prob

        # Pixels where rand_tensor >= salt_prob AND rand_tensor < total_prob are 'Pepper' (set to 0.0)
        pepper_mask = (rand_tensor >= self.salt_prob) & (rand_tensor < self.total_prob)

        # 3. Apply the noise using boolean indexing and broadcasting
        # The boolean mask (B, 1, H, W) is automatically broadcast to (B, C, H, W)
        # for assignment, which is highly efficient.

        batch[salt_mask.expand_as(batch)] = 1.0  # Apply Salt
        batch[pepper_mask.expand_as(batch)] = 0.0 # Apply Pepper

        return batch

    def __call__(self, inpt):
        # Ensures that a single image (3D) is treated as a batch (4D) for _apply_batch
        if isinstance(inpt, torch.Tensor):
            if inpt.dim() == 4:
                return self._apply_batch(inpt)
            elif inpt.dim() == 3:
                return self._apply_batch(inpt.unsqueeze(0)).squeeze(0)

        # Uses the default v2.Transform behavior for other types
        return super().__call__(inpt)