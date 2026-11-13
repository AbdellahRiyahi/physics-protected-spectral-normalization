"""New, explicit AICSET-240 revision implementation; not the historical fitter.

Inputs to every fit method must already be restricted to fitting groups. The
orchestrator owns partitions and must never pass calibration/test observations
to fitting, peak selection, imputation, rank selection, or hyperparameter tuning.
"""
from pathlib import Path
import json
import hashlib
import io
import os
import tempfile
import numpy as np
from scipy.signal import csd, find_peaks
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

LOG_RELATIVE_FLOOR = 1e-12
METHODS = ("raw", "signed_pca", "protected_signed", "protected_nnls")
IMPLEMENTATION_VERSION = "NEW_AICSET240_REVISION_V5_FULL_NOT_HISTORICAL_FITTER"


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be an integer >= {minimum}") from None
    if not np.isfinite(number) or number != int(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _grid(grid, bins, name="frequency grid"):
    values = np.asarray(grid, float)
    if (values.shape != (bins,) or bins < 3 or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(np.diff(values) <= 0)):
        raise ValueError(f"Invalid strictly increasing positive {name}")
    return values


def _spectrum(freq, sigma):
    f = np.asarray(freq, float); S = np.asarray(sigma, float)
    if f.ndim != 1 or len(f) < 3 or S.shape != (len(f), 2):
        raise ValueError("Expected frequency and sigma arrays with shape (n_frequency, 2)")
    if (not np.isfinite(f).all() or not np.isfinite(S).all()
            or np.any(f < 0) or np.any(np.diff(f) <= 0) or np.any(S < 0)):
        raise ValueError("Invalid finite nonnegative spectrum on increasing frequencies")
    return f, S


def _is_orthonormal(basis, atol=1e-9):
    return (basis.ndim == 2 and np.isfinite(basis).all()
            and np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=atol, rtol=0))


def _metadata_extra(metadata, extra):
    extra = {} if extra is None else dict(extra)
    for key in metadata.keys() & extra.keys():
        if _json_clean(extra[key]) != _json_clean(metadata[key]):
            raise ValueError(f"Extra metadata cannot override fitted model field {key!r}")
    return {**metadata, **extra}


def _matrix(X, name="X"):
    if np.iscomplexobj(X):
        raise ValueError(f"{name} must be real-valued")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError(f"{name} must be a nonempty finite two-dimensional array")
    return X


def _groups(groups, n):
    raw = np.asarray(groups, dtype=object)
    if raw.ndim != 1 or len(raw) != n or n < 1:
        raise ValueError("A nonmissing physical group identifier is required per row")
    g = np.array([str(value).strip() for value in raw])
    if np.any(np.isin(np.char.lower(g), ["", "nan", "none", "nat", "<na>"])):
        raise ValueError("A nonmissing physical group identifier is required per row")
    return g


def group_centroids(X, groups):
    X = _matrix(X); g = _groups(groups, len(X)); names = np.unique(g)
    return np.vstack([X[g == v].mean(axis=0) for v in names]), names


def group_weights(groups):
    """Weights sum to n, with equal total weight assigned to every group."""
    g = _groups(groups, len(groups)); names, counts = np.unique(g, return_counts=True)
    lookup = dict(zip(names, counts))
    return np.array([len(g) / (len(names) * lookup[v]) for v in g])


def cpsd_singular_spectra(signal, fs, nperseg=4096, overlap=0.5):
    """Welch CPSD within ONE synchronous acquisition; return f>0 and top 2 values.

    Uses periodic Hann, constant segment detrending, mean periodograms and
    one-sided density scaling. Short records and missing channels are rejected;
    segment length is never silently shortened. No filtering is repeated here.
    """
    Y = _matrix(signal, "signal"); nperseg = _positive_integer(nperseg, "nperseg", 8)
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Invalid fs, nperseg or overlap")
    n_overlap = int(np.floor(overlap * nperseg))
    if Y.shape[1] < 2 or len(Y) < nperseg + nperseg - n_overlap:
        raise ValueError("Need >=2 channels and >=2 Welch segments to support two singular curves")
    spectra = None
    for i in range(Y.shape[1]):
        for j in range(i, Y.shape[1]):
            f, value = csd(Y[:, i], Y[:, j], fs=float(fs), window="hann",
                           nperseg=nperseg, noverlap=n_overlap, detrend="constant",
                           return_onesided=True, scaling="density", average="mean")
            if spectra is None:
                spectra = np.zeros((len(f), Y.shape[1], Y.shape[1]), complex)
            spectra[:, i, j] = value
            spectra[:, j, i] = value.conj()
    # Each Welch matrix is Hermitian positive semidefinite. Its singular values
    # equal its nonnegative eigenvalues; tolerate only floating-point negativity.
    eigenvalues = np.linalg.eigvalsh(spectra)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -1e-9 * scale:
        raise ValueError("CPSD matrix is not positive semidefinite within tolerance")
    singular = np.maximum(eigenvalues[:, ::-1], 0.0)[:, :2]
    positive = f > 0
    if np.max(singular[positive, 1]) <= 64 * np.finfo(float).eps * np.max(singular[positive, 0]):
        raise ValueError("Selected channels provide no numerically supported second singular curve")
    return f[positive], singular[positive]


def grid_log_curves(freq, sigma, f1, upper=2.5, bins=128, lower=0.25,
                    relative_floor=LOG_RELATIVE_FLOOR):
    """Log spectra at f=f1*rho; floor each curve at max(curve)*relative_floor.

    Flooring is a NEW numerical convention, exported with the configuration.
    No extrapolation beyond the observed frequency support is permitted.
    """
    f, S = _spectrum(freq, sigma); bins = _positive_integer(bins, "bins", 8)
    if not np.isfinite(f1) or f1 <= 0 or not 0 < lower < upper or not np.isfinite(upper):
        raise ValueError("Invalid f1, spectrum, or normalized-frequency limits")
    if not np.isfinite(relative_floor) or not 0 < relative_floor < 1:
        raise ValueError("Invalid floor or grid size")
    target = float(f1) * np.linspace(lower, upper, bins)
    if target[0] < f[0] or target[-1] > f[-1]:
        raise ValueError("Requested normalized-frequency grid is outside observed support")
    curves = []
    for r in range(2):
        peak = float(S[:, r].max())
        if peak <= 0:
            raise ValueError(f"Retained singular curve {r + 1} is identically zero")
        floor = max(peak * relative_floor, np.finfo(float).tiny)
        curves.append(np.interp(target, f, np.log(np.maximum(S[:, r], floor))))
    return np.asarray(curves)


def normalize_curves(logcurves):
    """Per-curve frequency median centering and unit L2 norm, concatenated."""
    L = np.asarray(logcurves, float)
    single = L.ndim == 2
    if single: L = L[None, ...]
    if L.ndim != 3 or L.shape[1] != 2 or not len(L) or L.shape[-1] < 3 or not np.isfinite(L).all():
        raise ValueError("Expected (2,bins) or (n,2,bins) finite log curves")
    Z = L - np.median(L, axis=-1, keepdims=True)
    norms = np.linalg.norm(Z, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("At least one centered spectral curve has zero numerical norm")
    result = (Z / norms).reshape(len(L), -1)
    return result[0] if single else result


def estimate_reference_f1(training_spectra, band=(0.5, 20.0), prominence=0.1):
    """Explicit heuristic: first prominent peak of training median unit-height sigma1.

    Each list element is (frequency, singular_values). An audited physical f1
    override is preferable when this heuristic does not identify the first
    vertical mode. Returned audit data permit plotting and checking the choice.
    """
    records = [_spectrum(f, s) for f, s in training_spectra]
    if (not records or len(band) != 2 or not np.isfinite(band).all()
            or not 0 < band[0] < band[1] or not np.isfinite(prominence) or not 0 < prominence <= 1):
        raise ValueError("Need fitting spectra and valid f1 peak-selection settings")
    lo = max(float(band[0]), max(float(np.asarray(f)[0]) for f, _ in records))
    hi = min(float(band[1]), min(float(np.asarray(f)[-1]) for f, _ in records))
    f0 = np.asarray(records[0][0], float)
    grid = f0[(f0 >= lo) & (f0 <= hi)]
    if len(grid) < 8:
        raise ValueError("Insufficient common training spectral support for f1 selection")
    normalized = []
    for f, s in records:
        s = np.asarray(s, float)
        if s.ndim != 2 or s.shape[1] < 1: raise ValueError("Invalid fitting spectrum")
        curve = np.interp(grid, np.asarray(f, float), s[:, 0])
        if not np.isfinite(curve).all() or curve.max() <= 0:
            raise ValueError("Invalid first singular-value training spectrum")
        normalized.append(curve / curve.max())
    median = np.median(normalized, axis=0)
    peaks, _ = find_peaks(median, prominence=prominence * np.ptp(median))
    if not len(peaks):
        raise ValueError("No prominent fitting peak: supply an audited f1 override")
    f1 = float(grid[peaks[0]])
    return f1, {"selection": "FIRST_PROMINENT_TRAINING_MEDIAN_SIGMA1_PEAK",
                "band_hz": list(map(float, band)), "prominence_fraction": float(prominence),
                "frequency_hz": grid.tolist(), "median_unit_height_sigma1": median.tolist(),
                "candidate_peaks_hz": grid[peaks].tolist(), "selected_f1_hz": f1}


def shift_log_curves(logcurves, relative_shift, grid=None):
    """Translate in log frequency: L_shift(rho)=L(rho/(1+shift)).

    Constant endpoint extension is explicit and synthetic: it is not a raw-signal
    damage simulation. Root must report design shifts separately from new shifts.
    """
    L = np.asarray(logcurves, float)
    if L.ndim not in (2, 3) or L.shape[-2] != 2 or not np.isfinite(L).all():
        raise ValueError("Expected finite rank-two log curves")
    rho = _grid(np.linspace(0.25, 2.5, L.shape[-1]) if grid is None else grid, L.shape[-1])
    if not np.isfinite(relative_shift) or relative_shift <= -1:
        raise ValueError("relative_shift must exceed -1")
    log_rho = np.log(rho); query = log_rho - np.log1p(relative_shift)
    rows = L.reshape(-1, L.shape[-1])
    return np.stack([np.interp(query, log_rho, row, left=row[0], right=row[-1])
                     for row in rows]).reshape(L.shape)


def build_protected_basis(reference_log_curves, shifts=(-.02, -.01, -.005, .005, .01, .02),
                          grid=None, tolerance=1e-10):
    """Orthonormal span of per-rank finite-secants of the FITTING reference."""
    L = np.asarray(reference_log_curves, float)
    if L.ndim != 2 or L.shape[0] != 2: raise ValueError("Need reference shape (2,bins)")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Protection rank tolerance must be in (0,1)")
    base = normalize_curves(L); directions = []
    for rank in range(2):
        for shift in shifts:
            if shift == 0: continue
            shifted = L.copy()
            shifted[rank] = shift_log_curves(L, float(shift), grid=grid)[rank]
            directions.append(normalize_curves(shifted) - base)
    if not directions: raise ValueError("Need nonzero prescribed shifts")
    _, singular, Vt = np.linalg.svd(np.asarray(directions), full_matrices=False)
    count = int(np.sum(singular > tolerance * max(float(singular[0]), 1e-15)))
    if not count: raise ValueError("Prescribed shifts produce no protected direction")
    return Vt[:count].T


def _basis(vectors, rank, orientation=None, tolerance=1e-10):
    D = _matrix(vectors, "nuisance vectors")
    _, singular, Vt = np.linalg.svd(D, full_matrices=False)
    numerical_rank = int(np.sum(singular > tolerance * max(float(singular[0]), 1e-15)))
    if numerical_rank < rank:
        raise ValueError(f"Fixed nuisance rank {rank} unsupported; available rank={numerical_rank}")
    B = Vt[:rank].T.copy()
    reference = D.mean(axis=0) if orientation is None else np.asarray(orientation, float)
    for j in range(rank):
        product = float(B[:, j] @ reference)
        if abs(product) > tolerance * max(np.linalg.norm(reference), 1.0):
            sign = np.sign(product)
        else:
            sign = np.sign(B[np.argmax(np.abs(B[:, j])), j])
        B[:, j] *= sign if sign else 1.0
    return B, singular, numerical_rank


class SpectralNormalizer:
    """Frozen dictionary models; rank is fixed, never picked using evaluation data."""
    def __init__(self, rank=3):
        self.rank = _positive_integer(rank, "rank")

    def fit(self, X, groups, protected_basis=None, nuisance_vectors=None, rank=None):
        X = _matrix(X); self.rank = _positive_integer(self.rank if rank is None else rank, "rank")
        centroids, names = group_centroids(X, groups)
        self.mean_ = centroids.mean(axis=0); self.training_groups_ = names.tolist()
        D = centroids - self.mean_ if nuisance_vectors is None else _matrix(nuisance_vectors)
        if D.shape[1] != X.shape[1]: raise ValueError("Nuisance-vector dimensions mismatch")
        self.pca_basis_, self.pca_singular_values_, self.unprotected_rank_ = _basis(D, self.rank)
        self.U_ = np.empty((X.shape[1], 0))
        self.B_ = np.empty((X.shape[1], 0)); self.protected_singular_values_ = np.array([])
        self.available_methods_ = ["raw", "signed_pca"]
        if protected_basis is not None:
            U = _matrix(protected_basis, "protected basis")
            if U.shape[0] != X.shape[1] or not _is_orthonormal(U):
                raise ValueError("Protected basis must have orthonormal columns and matching rows")
            self.U_ = U.copy(); projected = D - (D @ U) @ U.T
            B, self.protected_singular_values_, self.protected_rank_ = _basis(projected, self.rank)
            # Remove tiny SVD roundoff in U, then reorthonormalize, keeping signs.
            projected_B = B - U @ (U.T @ B)
            Q, R = np.linalg.qr(projected_B)
            Q *= np.where(np.diag(R) < 0, -1.0, 1.0)
            self.B_ = Q[:, :self.rank]
            if np.max(np.abs(U.T @ self.B_)) > 1e-9:
                raise ArithmeticError("Protected-dictionary orthogonality failed")
            self.available_methods_ += ["protected_signed", "protected_nnls"]
        self.metadata_ = {"implementation": IMPLEMENTATION_VERSION,
                          "model_type": "SpectralNormalizer", "serialization_version": 1,
                          "rank": self.rank, "training_group_count": len(names),
                          "training_groups": self.training_groups_,
                          "mean": "EQUALLY_WEIGHTED_PHYSICAL_GROUP_CENTROIDS",
                          "nuisance_vectors": "EXPLICIT_TRAINING_VECTORS" if nuisance_vectors is not None
                          else "TRAINING_GROUP_CENTROID_MINUS_REFERENCE",
                          "orientation": "POSITIVE_MEAN_DISPLACEMENT_DOT_ELSE_LARGEST_LOADING_POSITIVE",
                          "protected_dimension": self.U_.shape[1]}
        return self

    def coefficients(self, X, method="protected_nnls"):
        X = _matrix(X)
        if method not in self.available_methods_: raise ValueError(f"Unavailable method {method}")
        if X.shape[1] != len(self.mean_): raise ValueError("Feature dimension mismatch")
        if method == "raw": return np.empty((len(X), 0))
        B = self.pca_basis_ if method == "signed_pca" else self.B_
        centered = X - self.mean_
        if method == "protected_nnls":
            return np.vstack([nnls(B, row, maxiter=max(30, 10 * self.rank))[0] for row in centered])
        return centered @ B

    def transform(self, X, method="protected_nnls"):
        X = _matrix(X); coefficients = self.coefficients(X, method)
        if method == "raw": return X.copy()
        B = self.pca_basis_ if method == "signed_pca" else self.B_
        result = X - coefficients @ B.T
        if method == "protected_nnls":
            if np.any(np.linalg.norm(result - self.mean_, axis=1) >
                      np.linalg.norm(X - self.mean_, axis=1) + 1e-9):
                raise ArithmeticError("NNLS residual norm increased")
        return result

    def score(self, X, method="protected_nnls"):
        return np.linalg.norm(self.transform(X, method) - self.mean_, axis=1)

    def save(self, prefix, extra=None):
        arrays = {"mean": self.mean_, "U": self.U_, "B": self.B_,
                  "pca_basis": self.pca_basis_, "pca_singular_values": self.pca_singular_values_,
                  "protected_singular_values": self.protected_singular_values_}
        return _save(prefix, arrays, _metadata_extra(self.metadata_, extra))

    @classmethod
    def load(cls, prefix):
        """Restore coefficients and bases without refitting or reading signals."""
        arrays, metadata = _load(prefix, "SpectralNormalizer")
        model = cls(metadata["rank"])
        model.mean_ = arrays["mean"]; model.U_ = arrays["U"]; model.B_ = arrays["B"]
        model.pca_basis_ = arrays["pca_basis"]
        model.pca_singular_values_ = arrays["pca_singular_values"]
        model.protected_singular_values_ = arrays["protected_singular_values"]
        if (model.mean_.ndim != 1 or model.pca_basis_.shape != (len(model.mean_), model.rank)
                or not _is_orthonormal(model.pca_basis_)):
            raise ValueError("Invalid spectral-normalizer reference or PCA basis")
        if model.U_.ndim != 2 or model.U_.shape[0] != len(model.mean_) or not _is_orthonormal(model.U_):
            raise ValueError("Invalid protected basis")
        model.available_methods_ = ["raw", "signed_pca"]
        if model.U_.shape[1]:
            if (model.B_.shape != (len(model.mean_), model.rank) or not _is_orthonormal(model.B_)
                    or np.max(np.abs(model.U_.T @ model.B_)) > 1e-9):
                raise ValueError("Invalid or non-protected nuisance dictionary")
            model.available_methods_ += ["protected_signed", "protected_nnls"]
        elif model.B_.shape != (len(model.mean_), 0):
            raise ValueError("Unprotected model cannot contain a protected dictionary")
        model.training_groups_ = list(metadata["training_groups"])
        model.metadata_ = metadata
        return model


class ConventionalPCA:
    """Conventional PCA residualization of ALL fitting acquisition features.

    The weighted covariance assigns equal total mass to each physical group.
    This differs from the centroid-displacement ``signed_pca`` ablation above:
    within-group acquisition variability participates in this PCA fit.
    """
    def __init__(self, rank=3):
        self.rank = _positive_integer(rank, "PCA rank")

    def fit(self, X, groups):
        X = _matrix(X); g = _groups(groups, len(X))
        weights = group_weights(g); weights = weights / weights.sum()
        self.mean_ = np.average(X, axis=0, weights=weights)
        weighted_centered = (X - self.mean_) * np.sqrt(weights[:, None])
        _, singular, Vt = np.linalg.svd(weighted_centered, full_matrices=False)
        numerical_rank = int(np.sum(singular > 1e-10 * max(float(singular[0]), 1e-15)))
        if numerical_rank < self.rank:
            raise ValueError(f"Fixed PCA rank {self.rank} unsupported; available rank={numerical_rank}")
        self.basis_ = Vt[:self.rank].T.copy()
        for j in range(self.rank):
            pivot = self.basis_[np.argmax(np.abs(self.basis_[:, j])), j]
            self.basis_[:, j] *= -1.0 if pivot < 0 else 1.0
        self.singular_values_ = singular
        self.explained_variance_ = singular[:self.rank] ** 2
        total = float(np.sum(singular ** 2))
        self.explained_variance_ratio_ = self.explained_variance_ / total
        self.training_groups_ = np.unique(g).tolist()
        self.n_training_acquisitions_ = len(X)
        return self

    def transform(self, X):
        X = _matrix(X)
        if X.shape[1] != len(self.mean_): raise ValueError("PCA feature dimension mismatch")
        return X - ((X - self.mean_) @ self.basis_) @ self.basis_.T

    def score(self, X):
        return np.linalg.norm(self.transform(X) - self.mean_, axis=1)

    def save(self, prefix, extra=None):
        arrays = {"mean": self.mean_, "basis": self.basis_,
                  "singular_values": self.singular_values_,
                  "explained_variance": self.explained_variance_,
                  "explained_variance_ratio": self.explained_variance_ratio_}
        metadata = {"implementation": IMPLEMENTATION_VERSION, "method": "CONVENTIONAL_WEIGHTED_PCA",
                    "model_type": "ConventionalPCA", "serialization_version": 1,
                    "rank": self.rank, "fitting_unit": "ALL_FITTING_ACQUISITIONS",
                    "weights": "EQUAL_TOTAL_WEIGHT_PER_PHYSICAL_GROUP",
                    "centering": "WEIGHTED_FITTING_ACQUISITION_MEAN", "whiten": False,
                    "training_groups": self.training_groups_,
                    "n_training_acquisitions": self.n_training_acquisitions_}
        return _save(prefix, arrays, _metadata_extra(metadata, extra))

    @classmethod
    def load(cls, prefix):
        """Restore this exact fitted PCA from its numeric NPZ and JSON export."""
        arrays, metadata = _load(prefix, "ConventionalPCA")
        model = cls(rank=metadata["rank"])
        model.mean_ = arrays["mean"]; model.basis_ = arrays["basis"]
        model.singular_values_ = arrays["singular_values"]
        model.explained_variance_ = arrays["explained_variance"]
        model.explained_variance_ratio_ = arrays["explained_variance_ratio"]
        if (model.mean_.ndim != 1 or model.basis_.shape != (len(model.mean_), model.rank)
                or not np.isfinite(model.mean_).all() or not np.isfinite(model.basis_).all()
                or not _is_orthonormal(model.basis_)):
            raise ValueError("Invalid or corrupted PCA fitted arrays")
        model.training_groups_ = metadata["training_groups"]
        model.n_training_acquisitions_ = int(metadata["n_training_acquisitions"])
        return model


class TemperatureRidge:
    """Conditional Ridge or one-temperature piecewise-linear Ridge baseline.

    CV preprocessors are fitted independently within each training fold. The
    piecewise model uses column zero only, with knots at fitting-group mean
    temperature quartiles. Missing covariates use fitting-only medians.
    """
    def __init__(self, piecewise=False, alpha_grid=(1e-4, 1e-3, 1e-2, .1, 1, 10, 100),
                 cv_splits=5, fixed_alpha=None):
        self.piecewise = bool(piecewise); self.alpha_grid = tuple(map(float, alpha_grid))
        self.cv_splits = _positive_integer(cv_splits, "cv_splits", 2); self.fixed_alpha = fixed_alpha
        if not self.alpha_grid or any(not np.isfinite(v) or v < 0 for v in self.alpha_grid):
            raise ValueError("Need >=2 CV splits and a nonempty predeclared alpha grid")

    def _fit_design(self, C, groups):
        C = C[:, :1] if self.piecewise else C
        median = np.nanmedian(C, axis=0)
        if not np.isfinite(median).all(): raise ValueError("A covariate is entirely missing in fitting data")
        filled = np.where(np.isnan(C), median, C)
        if not np.isfinite(filled).all(): raise ValueError("Infinite covariate values are forbidden")
        weights = group_weights(groups)
        mean = np.average(filled, axis=0, weights=weights)
        scale = np.sqrt(np.average((filled - mean) ** 2, axis=0, weights=weights))
        scale[scale < 1e-12] = 1.0
        gc, _ = group_centroids(filled, groups)
        knots = np.unique(np.quantile(gc[:, 0], [.25, .5, .75])) if self.piecewise else np.array([])
        return {"median": median, "mean": mean, "scale": scale, "knots": knots}

    def _design(self, covs, params):
        C = np.asarray(covs, float)
        if C.ndim == 1: C = C[:, None]
        if C.ndim != 2 or C.shape[1] != self.n_covariates_: raise ValueError("Covariate dimension mismatch")
        C = C[:, :1] if self.piecewise else C
        C = np.where(np.isnan(C), params["median"], C)
        if not np.isfinite(C).all(): raise ValueError("Invalid covariates")
        base = (C - params["mean"]) / params["scale"]
        if self.piecewise:
            hinges = np.maximum(C[:, :1] - params["knots"][None, :], 0) / params["scale"][0]
            return np.column_stack([base, hinges])
        return base

    def fit(self, X, covs, groups):
        X = _matrix(X); C = np.asarray(covs, float)
        if C.ndim == 1: C = C[:, None]
        if C.ndim != 2 or len(C) != len(X) or not C.shape[1]: raise ValueError("Invalid covariate array")
        if np.isinf(C).any(): raise ValueError("Infinite covariate values are forbidden")
        self.n_covariates_ = C.shape[1]; g = _groups(groups, len(X))
        self.mean_ = group_centroids(X, g)[0].mean(axis=0)
        self.cv_scores_ = []
        if self.fixed_alpha is None:
            if len(np.unique(g)) < 3: raise ValueError("Need >=3 training groups for grouped Ridge CV")
            splitter = GroupKFold(n_splits=min(self.cv_splits, len(np.unique(g))))
            for alpha in self.alpha_grid:
                if alpha < 0 or not np.isfinite(alpha): raise ValueError("Invalid Ridge penalty")
                errors = []
                for tr, va in splitter.split(X, groups=g):
                    params = self._fit_design(C[tr], g[tr])
                    model = Ridge(alpha=alpha).fit(self._design(C[tr], params), X[tr],
                                                   sample_weight=group_weights(g[tr]))
                    mse = np.mean((X[va] - model.predict(self._design(C[va], params))) ** 2, axis=1)
                    errors.extend(float(mse[g[va] == v].mean()) for v in np.unique(g[va]))
                self.cv_scores_.append({"alpha": alpha, "group_mean_mse": float(np.mean(errors))})
            self.alpha_ = min(self.cv_scores_, key=lambda v: (v["group_mean_mse"], v["alpha"]))["alpha"]
        else:
            self.alpha_ = float(self.fixed_alpha)
            if self.alpha_ < 0 or not np.isfinite(self.alpha_): raise ValueError("Invalid fixed penalty")
        self.params_ = self._fit_design(C, g)
        self.model_ = Ridge(alpha=self.alpha_).fit(self._design(C, self.params_), X,
                                                   sample_weight=group_weights(g))
        self.training_groups_ = np.unique(g).tolist()
        return self

    def predict(self, covs):
        return self.model_.predict(self._design(covs, self.params_))

    def transform(self, X, covs):
        X = _matrix(X)
        prediction = self.predict(covs)
        if X.shape != prediction.shape:
            raise ValueError("Covariates and spectral features must refer to the same rows and dimensions")
        return X - prediction + self.mean_

    def score(self, X, covs):
        return np.linalg.norm(self.transform(X, covs) - self.mean_, axis=1)

    def save(self, prefix, extra=None):
        arrays = {"mean": self.mean_, "coef": self.model_.coef_, "intercept": self.model_.intercept_,
                  **{f"cov_{k}": v for k, v in self.params_.items()}}
        metadata = {"implementation": IMPLEMENTATION_VERSION, "piecewise": self.piecewise,
                    "model_type": "TemperatureRidge", "serialization_version": 1,
                    "alpha": self.alpha_, "n_covariates": self.n_covariates_,
                    "alpha_grid": list(self.alpha_grid), "cv_splits": self.cv_splits,
                    "cv_scheme": "GROUP_K_FOLD_WITHIN_FITTING_PARTITION_ONLY",
                    "cv_scores": self.cv_scores_, "training_groups": self.training_groups_,
                    "knots": "FITTING_GROUP_MEAN_TEMPERATURE_QUARTILES" if self.piecewise else None,
                    "imputation_and_scaling": "FITTING_ONLY_PER_CV_FOLD_AND_FINAL_FIT"}
        return _save(prefix, arrays, _metadata_extra(metadata, extra))

    @classmethod
    def load(cls, prefix):
        arrays, metadata = _load(prefix, "TemperatureRidge")
        model = cls(piecewise=metadata["piecewise"], alpha_grid=metadata["alpha_grid"],
                    cv_splits=metadata["cv_splits"], fixed_alpha=metadata["alpha"])
        model.alpha_ = float(metadata["alpha"])
        model.n_covariates_ = _positive_integer(metadata["n_covariates"], "n_covariates")
        model.mean_ = arrays["mean"]
        model.params_ = {name: arrays["cov_" + name] for name in ("median", "mean", "scale", "knots")}
        used = 1 if model.piecewise else model.n_covariates_
        for name in ("median", "mean", "scale"):
            if model.params_[name].shape != (used,): raise ValueError("Invalid Ridge preprocessing arrays")
        if (np.any(model.params_["scale"] <= 0) or model.params_["knots"].ndim != 1
                or np.any(np.diff(model.params_["knots"]) <= 0)
                or (not model.piecewise and len(model.params_["knots"]))):
            raise ValueError("Invalid Ridge scales or knots")
        n_features = used + (len(model.params_["knots"]) if model.piecewise else 0)
        if (model.mean_.ndim != 1 or arrays["coef"].shape != (len(model.mean_), n_features)
                or arrays["intercept"].shape != model.mean_.shape):
            raise ValueError("Invalid Ridge regression coefficient dimensions")
        model.model_ = Ridge(alpha=model.alpha_)
        model.model_.coef_ = arrays["coef"]; model.model_.intercept_ = arrays["intercept"]
        model.model_.n_features_in_ = n_features
        model.training_groups_ = list(metadata["training_groups"])
        model.cv_scores_ = list(metadata["cv_scores"])
        return model


class FrequencyRatioBaseline:
    """Training-selected peak-band frequency ratios; a DIFFERENT feature family.

    Bands are frozen from the group-balanced training reference. By default two
    distinct peaks of sigma1 are tracked, avoiding duplicate resonances across
    sigma1 and sigma2. Ratios are spectral peak ratios, not identified physical
    modal parameters; their score units and feature dimension differ from spectra.
    Tracking uses a local concave quadratic when supported, otherwise a grid
    maximum. Band-edge maxima are reported, never silently discarded.
    """
    def __init__(self, grid=None, peaks_per_curve=2, prominence=0.05, curve_ranks=(0,)):
        self.grid = None if grid is None else np.asarray(grid, float)
        self.peaks_per_curve = _positive_integer(peaks_per_curve, "peaks_per_curve")
        self.prominence = float(prominence)
        ranks = tuple(curve_ranks)
        if (not ranks or len(set(ranks)) != len(ranks)
                or any(isinstance(r, (bool, np.bool_)) or r not in (0, 1) for r in ranks)
                or not np.isfinite(self.prominence) or not 0 < self.prominence <= 1):
            raise ValueError("Need unique curve ranks from {0,1} and prominence fraction in (0,1]")
        self.curve_ranks = tuple(sorted(int(r) for r in ranks))

    def fit(self, logcurves, groups=None):
        L = np.asarray(logcurves, float)
        if L.ndim != 3 or L.shape[1] != 2 or not len(L) or not np.isfinite(L).all():
            raise ValueError("Expected fitting log curves (n,2,bins)")
        g = np.arange(len(L)).astype(str) if groups is None else _groups(groups, len(L))
        self.grid_ = _grid(np.linspace(.25, 2.5, L.shape[-1]) if self.grid is None else self.grid,
                           L.shape[-1], "ratio frequency grid").copy()
        reference = group_centroids(L.reshape(len(L), -1), g)[0].mean(axis=0).reshape(L.shape[1:])
        self.bands_ = []; self.reference_curves_ = reference
        for rank in self.curve_ranks:
            peaks, properties = find_peaks(reference[rank], prominence=self.prominence * np.ptp(reference[rank]))
            if len(peaks) < self.peaks_per_curve:
                raise ValueError(f"Sigma{rank + 1} supports {len(peaks)} prominent fitting peaks; "
                                 f"predeclared baseline requires {self.peaks_per_curve}")
            strongest = np.argsort(-properties["prominences"], kind="stable")[:self.peaks_per_curve]
            selected = np.sort(peaks[strongest])
            boundaries = [0] + [int((a + b) // 2) for a, b in zip(selected[:-1], selected[1:])] + [L.shape[-1] - 1]
            for i, peak in enumerate(selected):
                self.bands_.append((rank, boundaries[i] + int(i > 0), boundaries[i + 1], int(peak)))
        if len(self.bands_) < 2:
            raise ValueError("Frequency-ratio baseline needs at least two identifiable fitting peak bands")
        self.n_bins_ = L.shape[-1]
        features = self.features(L)
        self.mean_ = group_centroids(features, g)[0].mean(axis=0)
        weights = group_weights(g)
        self.scale_ = np.sqrt(np.average((features - self.mean_) ** 2, axis=0, weights=weights))
        self.constant_features_ = np.flatnonzero(self.scale_ <= 1e-12).tolist()
        self.scale_[self.scale_ <= 1e-12] = 1.0
        self.training_groups_ = np.unique(g).tolist()
        return self

    def tracked_frequencies(self, logcurves):
        """Return rho estimates and per-band boundary/fit diagnostics per row."""
        L = np.asarray(logcurves, float)
        if L.ndim == 2: L = L[None, ...]
        if L.ndim != 3 or not len(L) or L.shape[1:] != (2, len(self.grid_)) or not np.isfinite(L).all():
            raise ValueError("Invalid ratio-input log curves")
        values = np.empty((len(L), len(self.bands_)))
        edges = np.zeros(values.shape, bool); interpolated = np.zeros(values.shape, bool)
        for j, (rank, lo, hi, _) in enumerate(self.bands_):
            indices = lo + np.argmax(L[:, rank, lo:hi + 1], axis=1)
            values[:, j] = self.grid_[indices]
            edges[:, j] = (indices == lo) | (indices == hi)
            for row, k in enumerate(indices):
                if k <= lo or k >= hi: continue
                x = self.grid_[k - 1:k + 2] - self.grid_[k]
                y = L[row, rank, k - 1:k + 2]
                a, b, _ = np.polyfit(x, y - y[1], 2)
                if a >= 0: continue
                offset = -b / (2 * a)
                if not x[0] <= offset <= x[2]: continue
                values[row, j] += offset; interpolated[row, j] = True
        return values, {"band_edge_maximum": edges, "quadratic_interpolation_used": interpolated}

    def features(self, logcurves):
        values, _ = self.tracked_frequencies(logcurves)
        # First selected band supplies an acquisition-specific denominator.
        return values[:, 1:] / values[:, :1]

    def transform(self, logcurves):
        return (self.features(logcurves) - self.mean_) / self.scale_

    def score(self, logcurves):
        return np.linalg.norm(self.transform(logcurves), axis=1)

    def save(self, prefix, extra=None):
        arrays = {"grid": self.grid_, "bands": np.asarray(self.bands_, int),
                  "reference_curves": self.reference_curves_, "mean": self.mean_, "scale": self.scale_}
        metadata = {"implementation": IMPLEMENTATION_VERSION, "family": "SPECTRAL_PEAK_FREQUENCY_RATIOS_DIFFERENT_DIMENSION",
                    "model_type": "FrequencyRatioBaseline", "serialization_version": 1,
                    "peaks_per_curve": self.peaks_per_curve, "prominence_fraction": self.prominence,
                    "curve_ranks_zero_based": list(self.curve_ranks),
                    "peak_tracking": "LOCAL_CONCAVE_QUADRATIC_ELSE_GRID_MAXIMUM; BAND_EDGES_FLAGGED",
                    "physical_modal_identification": False,
                    "training_groups": self.training_groups_, "constant_features_scale_set_to_one": self.constant_features_}
        return _save(prefix, arrays, _metadata_extra(metadata, extra))

    @classmethod
    def load(cls, prefix):
        arrays, metadata = _load(prefix, "FrequencyRatioBaseline")
        grid = arrays["grid"]
        if grid.ndim != 1: raise ValueError("Invalid peak-ratio grid")
        grid = _grid(grid, len(grid))
        model = cls(grid=grid, peaks_per_curve=metadata["peaks_per_curve"],
                    prominence=metadata["prominence_fraction"], curve_ranks=metadata["curve_ranks_zero_based"])
        bands = arrays["bands"]
        if bands.ndim != 2 or bands.shape[1] != 4 or len(bands) < 2 or not np.equal(bands, np.floor(bands)).all():
            raise ValueError("Invalid peak-ratio bands")
        bands = bands.astype(int)
        for rank, lo, hi, peak in bands:
            if rank not in model.curve_ranks or not 0 <= lo <= peak <= hi < len(grid) or lo >= hi:
                raise ValueError("Invalid peak-ratio band limits")
        if (arrays["reference_curves"].shape != (2, len(grid))
                or arrays["mean"].shape != (len(bands) - 1,)
                or arrays["scale"].shape != arrays["mean"].shape or np.any(arrays["scale"] <= 0)):
            raise ValueError("Invalid peak-ratio reference or scaling arrays")
        model.grid_ = grid; model.n_bins_ = len(grid); model.bands_ = [tuple(v) for v in bands.tolist()]
        model.reference_curves_ = arrays["reference_curves"]; model.mean_ = arrays["mean"]
        model.scale_ = arrays["scale"]; model.training_groups_ = list(metadata["training_groups"])
        model.constant_features_ = list(metadata["constant_features_scale_set_to_one"])
        return model


def _json_clean(value):
    if isinstance(value, dict): return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray): return _json_clean(value.tolist())
    if isinstance(value, np.generic): return _json_clean(value.item())
    if isinstance(value, Path): return str(value)
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError(f"Unsupported metadata value type {type(value).__name__}")


def _model_paths(prefix):
    prefix = Path(prefix)
    if prefix.suffix in (".npz", ".json"): prefix = prefix.with_suffix("")
    return Path(str(prefix) + ".npz"), Path(str(prefix) + ".json")


def _numeric_arrays(arrays):
    result = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Only finite real numeric arrays are allowed in fitted model exports")
    return result


def _save(prefix, arrays, metadata):
    npz, meta = _model_paths(prefix)
    arrays = _numeric_arrays(arrays)
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays); data = stream.getvalue()
    metadata = {**_json_clean(metadata), "arrays_sha256": hashlib.sha256(data).hexdigest()}
    text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Prepare both files before replacing either. A half-completed replacement
    # cannot be mistaken for a matching pair because reload verifies the hash.
    npz.parent.mkdir(parents=True, exist_ok=True); temporary = []
    try:
        for target, content in ((npz, data), (meta, text.encode("utf-8"))):
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", delete=False) as file:
                file.write(content); temporary.append((Path(file.name), target))
        for source, target in temporary: os.replace(source, target)
    finally:
        for source, _ in temporary:
            if source.exists(): source.unlink()
    return {"arrays": str(npz), "metadata": str(meta)}


def _load(prefix, expected_type):
    npz, meta = _model_paths(prefix)
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    if (metadata.get("model_type") != expected_type or metadata.get("serialization_version") != 1
            or metadata.get("implementation") != IMPLEMENTATION_VERSION):
        raise ValueError(f"Expected a V3 {expected_type} fitted model export")
    data = npz.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata.get("arrays_sha256"):
        raise ValueError("Fitted array archive does not match its JSON metadata SHA-256")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = _numeric_arrays({name: archive[name].copy() for name in archive.files})
    return arrays, metadata


def export_model(model, prefix, extra=None):
    return model.save(prefix, extra=extra)
