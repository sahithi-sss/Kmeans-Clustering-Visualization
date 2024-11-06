import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from tkinter import Tk, Button, Scale, Frame, Label, HORIZONTAL, LEFT, RIGHT
from PIL import Image, ImageTk
import threading
import matplotlib
matplotlib.use("Agg")

# Initialize data
n_samples = 500
n_features = 2
data, _ = make_blobs(n_samples=n_samples, centers=2, n_features=n_features, random_state=42)
centroids_history = []

def initialize_kmeans(K):
    global data, centroids_history
    data, _ = make_blobs(n_samples=n_samples, centers=K, n_features=n_features, random_state=42)
    centroids = data[np.random.choice(data.shape[0], K, replace=False)]
    centroids_history = [centroids.copy()]

    max_iters = 20 if K > 5 else 10
    for i in range(1, max_iters):
        labels = np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data])
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(K)])
        centroids_history.append(new_centroids.copy())
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

def create_animation(K, gif_path="kmeans_animation.gif"):
    initialize_kmeans(K)
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.get_cmap('tab10', K)

    def update(frame):
        ax.clear()
        centroids = centroids_history[min(frame, len(centroids_history) - 1)]
        labels = np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data])

        for k in range(K):
            cluster_points = data[labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=colors(k), alpha=0.6)
            ax.scatter(centroids[k, 0], centroids[k, 1], color='black', marker='x', s=100)
        ax.set_title(f"K-means Iteration {frame + 1} for K={K}")
    
    anim = FuncAnimation(fig, update, frames=len(centroids_history), interval=3000)
    anim.save(gif_path, writer='pillow')
    plt.close(fig)

class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-means Clustering Visualization")
        self.root.configure(bg="black")

        # Left control panel
        control_frame = Frame(self.root, bg="black")
        control_frame.pack(side=LEFT, padx=10, pady=10)

        # Slider for number of clusters K
        self.k_slider = Scale(control_frame, from_=2, to=10, orient=HORIZONTAL, label="Number of Clusters (K)", bg="black", fg="white")
        self.k_slider.pack(pady=10)
        
        # Refresh button
        self.refresh_button = Button(control_frame, text="Refresh", command=self.update_animation)
        self.refresh_button.pack(pady=10)

        # Progress bar
        self.progress_label = Label(control_frame, text="Progress: 0%", bg="black", fg="white")
        self.progress_label.pack(pady=10)

        # GIF display label
        self.gif_label = Label(self.root, bg="black")
        self.gif_label.pack(side=RIGHT, padx=10, pady=10)

        # Initialize latest K tracker to prevent "dancing"
        self.current_k = self.k_slider.get()

        # Generate initial animation
        self.update_animation()

    def update_animation(self):
        # Ensure only the latest K is rendered
        self.current_k = self.k_slider.get()
        K = self.current_k
        self.progress_label.config(text="Progress: 0%")

        def run_animation():
            create_animation(K, gif_path="kmeans_animation.gif")
            if self.current_k == K:  # Render only if no slider changes occurred
                self.display_gif("kmeans_animation.gif")

        # Run animation in a separate thread
        threading.Thread(target=run_animation).start()

    def display_gif(self, gif_path):
        gif = Image.open(gif_path)
        frames = []

        try:
            while True:
                frame = gif.copy()
                frames.append(ImageTk.PhotoImage(frame))
                gif.seek(len(frames))
        except EOFError:
            pass

        def update_frame(index):
            # If user changed K, stop updating the GIF to avoid conflicts
            if self.current_k != self.k_slider.get():
                return
            self.gif_label.config(image=frames[index])
            progress_percent = int((index + 1) / len(frames) * 100)
            self.progress_label.config(text=f"Progress: {progress_percent}%")
            self.root.after(3000, update_frame, (index + 1) % len(frames))

        update_frame(0)

# Main execution
root = Tk()
app = KMeansApp(root)
root.mainloop()
