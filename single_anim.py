import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.datasets import make_blobs

# General parameters
n_samples = 500  # Total points per clustering
n_features = 2   # Points in 2D space

# Updated values of K
K = 7  # Single value for K

# Function to initialize and animate K-means clustering
def animate_kmeans(K, max_iters=20):
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f"K-means Clustering with K={K}")

    # Initialize colors for clusters
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Generate data and initialize centroids history
    data, _ = make_blobs(n_samples=n_samples, centers=K, n_features=n_features, random_state=42)
    centroids = data[np.random.choice(data.shape[0], K, replace=False)]
    centroids_history = [centroids.copy()]

    # Perform K-means to get centroid positions for each iteration
    for i in range(1, max_iters):
        # Assign points to the nearest centroid
        labels = np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data])

        # Recompute centroids as the mean of each cluster
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(K)])
        centroids_history.append(new_centroids.copy())

        # Stop if centroids do not change
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    def update(frame):
        ax.clear()  # Clear the plot for each frame

        # Get the centroids for the current frame
        current_centroids = centroids_history[min(frame, len(centroids_history) - 1)]
        labels = np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in current_centroids]) for point in data])

        # Plot each cluster with its points
        for k in range(K):
            cluster_points = data[labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=colors[k % len(colors)], alpha=0.6, label=f"Cluster {k+1}")
            ax.scatter(current_centroids[k, 0], current_centroids[k, 1], color='black', marker='x', s=100)  # Plot centroid

        ax.set_title(f"K-means Iteration {frame + 1} for K={K}")
        ax.legend(loc="upper right")

    # Create the animation with a 2-second interval between frames
    anim = FuncAnimation(fig, update, frames=max_iters, interval=2000, repeat=False)
    plt.tight_layout()  # Adjust layout to fit titles
    plt.show()

    # Save the animation as a GIF
    anim.save('kmeans_animation_single_k.gif', writer='pillow', fps=0.5)  # 0.5 fps for 2 seconds per frame

    return anim

# Run the animation
animate_kmeans(K, max_iters=20)
