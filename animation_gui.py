import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.datasets import make_blobs

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tkinter GUI with K-Means Clustering Animation")
        self.geometry("800x600")

        # Create the plot area
        self.plot_frame = tk.Frame(self, width=450, height=600)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = plt.figure(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create the slider and button area
        self.control_frame = tk.Frame(self, width=350, height=600, bg='grey')
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        self.k_label = tk.Label(self.control_frame, text="K Value", bg='grey', fg='white')
        self.k_label.pack(pady=10)

        self.k_value = tk.IntVar(value=2)  
        self.k_slider = ttk.Scale(self.control_frame, from_=1, to=10, variable=self.k_value, orient=tk.HORIZONTAL)
        self.k_slider.pack(pady=10)
        self.k_slider.bind("<ButtonRelease-1>", self.update_plot)

        # Add display for current K value
        self.k_display = tk.Label(self.control_frame, 
                                text=f"Current K: {self.k_value.get()}", 
                                bg='grey', 
                                fg='white',
                                font=('Arial', 12, 'bold'))
        self.k_display.pack(pady=5)
        
        # Update k_display whenever the slider value changes
        self.k_value.trace('w', self.update_k_display)

        self.refresh_button = tk.Button(self.control_frame, text="Refresh", command=self.restart_animation, bg='grey', fg='white')
        self.refresh_button.pack(pady=10)

        self.animation = None
        self.update_plot(None)

    def update_k_display(self, *args):
        self.k_display.config(text=f"Current K: {self.k_value.get()}")

    def animate_kmeans(self, K, max_iters=20):
        # Set up the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        fig = self.figure
        fig.set_facecolor('black') 

        # Initialize colors for clusters
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']

        # Generate data and initialize centroids history
        n_samples = 500
        n_features = 2
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
            ax.clear()  

            # Get the centroids for the current frame
            current_centroids = centroids_history[min(frame, len(centroids_history) - 1)]
            labels = np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in current_centroids]) for point in data])

            # Plot each cluster with its points
            for k in range(K):
                cluster_points = data[labels == k]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=colors[k % len(colors)], alpha=0.6, label=f"Cluster {k+1}")
                ax.scatter(current_centroids[k, 0], current_centroids[k, 1], color='black', marker='x', s=100)  # Plot centroid

                ax.set_title(f"K-means Iteration {frame + 1} for K={K}", color='white')

            ax.legend(loc="upper right")

        # Create the animation with a 2-second interval between frames
        self.animation = FuncAnimation(fig, update, frames=max_iters, interval=2000, repeat=False)
        plt.tight_layout()  
        return self.animation

    def update_plot(self, event):
        K = self.k_value.get()
        #print(f"K value: {K}")
        if K > 0:
            if self.animation:
                self.animation.event_source.stop()
            self.animation = self.animate_kmeans(K, max_iters=20)
            self.canvas.draw()
            self.animation.save('kmeans_animation_single_k.gif', writer='pillow', fps=0.5)
        else:
            print("K value must be greater than 0.")

    def restart_animation(self):
        self.update_plot(None)

if __name__ == "__main__":
    app = App()
    app.mainloop()