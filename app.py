import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_digits
from sklearn.preprocessing import StandardScaler

# ==========================================
# K-MEANS FROM SCRATCH
# ==========================================
class KMeansFromScratch:
    def __init__(self, k=10, max_iters=100, init_method='k-means++'):
        self.k = k
        self.max_iters = max_iters
        self.init_method = init_method
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        n_samples = X.shape[0]

        if self.init_method == 'random':
            indices = np.random.choice(n_samples, self.k, replace=False)
            return X[indices]

        elif self.init_method == 'k-means++':
            centroids = [X[np.random.choice(n_samples)]]

            for _ in range(1, self.k):
                dists = np.array([
                    min([np.sum((x - c) ** 2) for c in centroids])
                    for x in X
                ])

                probs = dists / np.sum(dists)
                cumulative_probs = np.cumsum(probs)
                r = np.random.rand()

                for idx, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids.append(X[idx])
                        break

            return np.array(centroids)

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            new_labels = np.argmin(distances, axis=0)

            if self.labels is not None and np.array_equal(new_labels, self.labels):
                break

            self.labels = new_labels

            new_centroids = []
            for i in range(self.k):
                points = X[self.labels == i]
                if len(points) > 0:
                    new_centroids.append(points.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[i])

            self.centroids = np.array(new_centroids)

    def get_inertia(self, X):
        inertia = 0
        for i in range(self.k):
            points = X[self.labels == i]
            if len(points) > 0:
                inertia += np.sum((points - self.centroids[i]) ** 2)
        return inertia


# ==========================================
# EXPERIMENT FUNCTION
# ==========================================
def run_comparison(X, k, n_trials=30, title="Experiment"):
    random_results = []
    plus_results = []

    print(f"\nRunning {n_trials} trials for {title}...")

    for i in range(n_trials):
        km_r = KMeansFromScratch(k=k, init_method='random')
        km_r.fit(X)
        random_results.append(km_r.get_inertia(X))

        km_p = KMeansFromScratch(k=k, init_method='k-means++')
        km_p.fit(X)
        plus_results.append(km_p.get_inertia(X))

  # ==================================
# HISTOGRAM + BOXPLOT (SUBPLOT)
# ==================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # HISTOGRAM
    axes[0].hist(random_results, bins=20, alpha=0.6, label='Random Init', edgecolor='black')
    axes[0].hist(plus_results, bins=20, alpha=0.6, label='K-means++', edgecolor='black')
    axes[0].set_title(f'Distribution: {title}')
    axes[0].set_xlabel('Inertia')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # BOXPLOT
    axes[1].boxplot([random_results, plus_results], labels=['Random', 'K-means++'])
    axes[1].set_title(f'Variability: {title}')
    axes[1].set_ylabel('Inertia')

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=150)
    plt.show()

    # ==================================
    # NUMERIC SUMMARY
    # ==================================
    print(f"\n{title} Results:")
    print(f"Random Init -> Mean: {np.mean(random_results):.2f}, Std: {np.std(random_results):.2f}")
    print(f"K-means++ -> Mean: {np.mean(plus_results):.2f}, Std: {np.std(plus_results):.2f}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    # ========= SYNTHETIC =========
    print("\n--- SYNTHETIC DATA ---")
    X_syn, _ = make_blobs(
        n_samples=1000,
        centers=8,
        cluster_std=2.5,
        random_state=42
    )

    run_comparison(X_syn, k=8, n_trials=50, title="Synthetic Dataset")

    # ========= MNIST =========
    print("\n--- MNIST DATASET ---")
    print("Loading MNIST...")

    digits = load_digits()

X_mnist = digits.data

X_mnist = StandardScaler().fit_transform(X_mnist)

run_comparison(X_mnist, k=10, n_trials=10, title="MNIST Dataset")