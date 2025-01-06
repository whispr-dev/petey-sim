import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MultiLayerFuzzyForestFireCA:
    """
    Multi-layer fuzzy forest-fire CA with different parameters per layer.
    
    Each cell is a float in [0,1]:
      - ~0.0 => empty
      - (0,1) => tree-ish
      - 1.0   => burning

    We store parameters for each layer separately:
      - p[l]          => Probability of tree growth in empty cell (layer l)
      - f[l]          => Probability of spontaneous ignition in a tree cell (layer l)
      - burn_rate[l]  => How fast burning cells burn down
      - spread_factor[l] => How strongly burning neighbors ignite this cell
    """

    def __init__(self, layers=3, width=40, height=40,
                 p=None, f=None, burn_rate=None, spread_factor=None):
        """
        p, f, burn_rate, spread_factor should be lists or arrays of length `layers`,
        each specifying the parameter for that layer.

        If not provided, we’ll just pick some default “flavors” for top/middle/bottom.
        """
        self.layers = layers
        self.width = width
        self.height = height

        # If user didn't supply parameter arrays, define some interesting defaults
        if p is None:
            # e.g. top=windy => lower tree growth, middle=dense => higher growth, bottom=moist => moderate
            p = [0.01, 0.03, 0.02]
        if f is None:
            # e.g. top=windy => higher chance of ignition, middle=average, bottom=moist => lower chance
            f = [0.001, 0.0005, 0.0001]
        if burn_rate is None:
            # e.g. top=windy => faster burn, middle=moderate, bottom=slow burn
            burn_rate = [0.4, 0.2, 0.1]
        if spread_factor is None:
            # e.g. top=windy => stronger spread, middle=moderate, bottom=weaker
            spread_factor = [0.5, 0.3, 0.1]

        # Convert to numpy arrays for convenience
        self.p = np.array(p, dtype=float)
        self.f = np.array(f, dtype=float)
        self.burn_rate = np.array(burn_rate, dtype=float)
        self.spread_factor = np.array(spread_factor, dtype=float)

        # 3D grid: shape (layers, height, width)
        self.grid = np.random.rand(layers, height, width)

        # Optionally set some random fraction to exactly 1.0 to represent initial burning
        burning_mask = (np.random.rand(layers, height, width) < 0.02)  # 2% burning start
        self.grid[burning_mask] = 1.0

    def update(self):
        new_grid = np.copy(self.grid)

        # We’ll iterate over each cell in each layer
        for l in range(self.layers):
            for r in range(self.height):
                for c in range(self.width):
                    cell_val = self.grid[l, r, c]

                    # We'll gather neighbor burning from the 3D neighborhood
                    neighbor_burning = 0.0
                    count = 0
                    for dl in [-1, 0, 1]:
                        ll = l + dl
                        if 0 <= ll < self.layers:
                            for dr in [-1, 0, 1]:
                                rr = (r + dr) % self.height
                                for dc in [-1, 0, 1]:
                                    cc = (c + dc) % self.width
                                    if not (dl == 0 and dr == 0 and dc == 0):
                                        neighbor_burning += self.grid[ll, rr, cc]
                                        count += 1

                    # The fraction of neighbors that are burning
                    # (In the 3D Moore neighborhood, max is 26 neighbors if layers=3, fewer if layer=1 or 2)
                    burning_fraction = neighbor_burning / count if count > 0 else 0.0

                    # If the cell is burning
                    if cell_val >= 0.99:
                        # burn it down by burn_rate[l]
                        new_val = cell_val - self.burn_rate[l]
                        if new_val < 0.0:
                            new_val = 0.0
                        new_grid[l, r, c] = new_val
                    else:
                        # Possibly spontaneously ignite
                        if cell_val > 0.1 and random.random() < self.f[l]:
                            # jump cell value near 1.0
                            new_grid[l, r, c] = min(1.0, cell_val + 0.9)

                        # Possibly spontaneously grow if empty
                        elif cell_val < 0.1 and random.random() < self.p[l]:
                            # jump up to moderate tree
                            new_grid[l, r, c] = 0.3 + 0.2 * random.random()
                        else:
                            # Fire spread from neighbors
                            # We push up by a fraction of spread_factor[l] * burning_fraction
                            spread_push = burning_fraction * self.spread_factor[l]
                            new_val = cell_val + spread_push
                            if new_val > 1.0:
                                new_val = 1.0
                            new_grid[l, r, c] = new_val

        self.grid = new_grid

    def inject_inputs(self, inputs):
        """
        If you want to integrate your fuzzy neuron layer or external signals, 
        you could interpret them in some meaningful way here.
        For now, do nothing or do something minimal, e.g. randomly nudge cells.
        """
        for val in inputs:
            l = random.randint(0, self.layers - 1)
            r = random.randint(0, self.height - 1)
            c = random.randint(0, self.width - 1)

            # Example approach: if val > 0.7 => partial ignition, else grow
            if val > 0.7:
                self.grid[l, r, c] = min(1.0, self.grid[l, r, c] + 0.5)
            else:
                self.grid[l, r, c] = max(0.1, self.grid[l, r, c])

    def get_average_value(self):
        return float(np.mean(self.grid))

# -----------------------------------------------------
# Animation Demo
# -----------------------------------------------------

def update_frame(frame_index, ca_obj, images, text):
    """
    Called by FuncAnimation each frame. We:
      1) Update the CA by one step
      2) Update each layer's imshow data
      3) Update text about average cell value
    """
    ca_obj.update()
    avg_val = ca_obj.get_average_value()

    for l, im in enumerate(images):
        im.set_array(ca_obj.grid[l])
    text.set_text(f"Step {frame_index+1}, Avg: {avg_val:.3f}")

    return images + [text]


if __name__ == "__main__":
    # Initialize our forest-fire CA with layer-specific "flavors"
    # The defaults inside the constructor are:
    #  top (layer 0) => windy => p=0.01, f=0.001, burn_rate=0.4, spread_factor=0.5
    #  middle (layer 1) => dense => p=0.03, f=0.0005, burn_rate=0.2, spread_factor=0.3
    #  bottom (layer 2) => moist => p=0.02, f=0.0001, burn_rate=0.1, spread_factor=0.1
    ca = MultiLayerFuzzyForestFireCA(layers=3, width=50, height=50)

    # Set up matplotlib figure
    fig, axes = plt.subplots(1, ca.layers, figsize=(12, 4))
    images = []
    for l in range(ca.layers):
        ax = axes[l]
        # Use a color map; we'll limit it to [0,1]
        im = ax.imshow(ca.grid[l], cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f"Layer {l}")
        ax.axis('off')
        images.append(im)

    # We'll put a text label on the first axis to show step and average
    text = axes[0].text(0.02, 1.05, "", transform=axes[0].transAxes, color="white")

    # Create the animation
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        fargs=(ca, images, text),   # arguments passed to update_frame
        frames=200,                 # number of frames (simulation steps)
        interval=200,               # time in ms between frames
        blit=False                  # we can try True for better performance, but might cause artifacts
    )

    plt.tight_layout()
    plt.show()
