import torch


def compute_correlation_loss(X_distances: torch.Tensor, Z_distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the negative Pearson correlation between distances in input and embedding spaces.

    Returns negative correlation (minimize this to maximize correlation).
    Returns 0.0 if either distance vector has zero variance (e.g. at initialization).

    Parameters
    ----------
    X_distances : torch.Tensor
        Distances in input space, shape (batch_size,)
    Z_distances : torch.Tensor
        Distances in embedding space, shape (batch_size,)
    eps : float
        Small constant to detect near-zero variance

    Returns
    -------
    torch.Tensor
        Negative Pearson correlation coefficient
    """
    X_centered = X_distances - X_distances.mean()
    Z_centered = Z_distances - Z_distances.mean()

    X_var = (X_centered ** 2).mean()
    Z_var = (Z_centered ** 2).mean()

    # If either has ~zero variance, correlation is undefined — return 0 (no gradient signal)
    if X_var < eps or Z_var < eps:
        return torch.tensor(0.0, device=X_distances.device, dtype=X_distances.dtype)

    numerator = (X_centered * Z_centered).mean()
    correlation = numerator / (torch.sqrt(X_var) * torch.sqrt(Z_var))

    return -correlation
