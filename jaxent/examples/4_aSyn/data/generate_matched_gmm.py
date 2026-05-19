import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
from pathlib import Path
from scipy.stats import mode

tris_dir = Path("jaxent/examples/4_aSyn/data/_cluster_inertia")
shape_axes = np.load(tris_dir / "shape_axes.npy")
original_labels = np.load(tris_dir / "cluster_labels.npy")

gmm = GaussianMixture(n_components=20, covariance_type="full", random_state=42)
gmm.fit(shape_axes)
new_labels = gmm.predict(shape_axes)

mapping = {}
for i in range(20):
    mask = new_labels == i
    if np.any(mask):
        m = mode(original_labels[mask], keepdims=True).mode[0]
        mapping[i] = m

print("Mapping new -> original:", mapping)

new_weights = np.zeros_like(gmm.weights_)
new_means = np.zeros_like(gmm.means_)
new_covariances = np.zeros_like(gmm.covariances_)
new_precisions = np.zeros_like(gmm.precisions_)
new_precisions_cholesky = np.zeros_like(gmm.precisions_cholesky_)

for new_id, orig_id in mapping.items():
    new_weights[orig_id] = gmm.weights_[new_id]
    new_means[orig_id] = gmm.means_[new_id]
    new_covariances[orig_id] = gmm.covariances_[new_id]
    new_precisions[orig_id] = gmm.precisions_[new_id]
    new_precisions_cholesky[orig_id] = gmm.precisions_cholesky_[new_id]

gmm.weights_ = new_weights
gmm.means_ = new_means
gmm.covariances_ = new_covariances
gmm.precisions_ = new_precisions
gmm.precisions_cholesky_ = new_precisions_cholesky

test_labels = gmm.predict(shape_axes)
match_ratio = np.mean(test_labels == original_labels)
print(f"Match ratio after reordering: {match_ratio:.4f}")

if match_ratio > 0.99:
    joblib.dump(gmm, tris_dir / "gmm_model.pkl")
    print("Saved reordered GMM to gmm_model.pkl")
else:
    print("Warning: Match ratio is too low, check mapping!")
