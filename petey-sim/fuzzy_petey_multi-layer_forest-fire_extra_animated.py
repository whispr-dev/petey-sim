import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MultiLayerFuzzyForestFireCA:
    """
    Multi-layer fuzzy forest-fire CA with different parameters per layer, plus:
      - Time-varying ignition (flip every 20 steps).
      - Column-based dryness factor (cells on the right are drier).
      - Wind from the east (neighbors to the west have double weight).
    Each cell: float in [0,1] -> 0=empty, (0,1)=tree, 1=burning.
    """

    def __init__(self, layers=3, width=40, height=40,
                 p=None, f=None, burn_rate=None, spread_factor=None):
        """
        p, f, burn_rate, spread_factor: lists of length = layers
        If None, default flavors for top/middle/bottom.
        """
        self.layers = layers
        self.width = width
        self.height = height

        # Some "default flavors"
        if p is None:
            p = [0.01, 0.03, 0.02]      # top=low growth, mid=high, bottom=med
        if f is None:
            f = [0.001, 0.0005, 0.0001] # top=windy => higher ignition, etc.
        if burn_rate is None:
            burn_rate = [0.5, 0.3, 0.2]
        if spread_factor is None:
            spread_factor = [0.6, 0.35, 0.15]

        # Store as numpy arrays
        self.p = np.array(p, dtype=float)
        self.f = np.array(f, dtype=float)
        self.burn_rate = np.array(burn_rate, dtype=float)
        self.spread_factor = np.array(spread_factor, dtype=float)

        # 3D grid: shape = (layers, height, width)
        self.grid = np.random.rand(layers, height, width)

        # 2% burning start
        burning_mask = (np.random.rand(layers, height, width) < 0.02)
        self.grid[burning_mask] = 1.0

        # Keep track of simulation steps
        self.step_count = 0

    def update(self):
        """
        Advance the CA one step, applying:
          1) Time-based changes to f[l].
          2) Column-based dryness factor to scale f[l] and spread_factor[l].
          3) Wind weighting in neighbor summation.
        """
        # 1) Time-based parameter changes (flip every 20 steps)
        time_factor = (self.step_count // 20) % 2  # 0 or 1, flips every 20 steps
        for l in range(self.layers):
            # Example: let baseline f[l] jump between two values every 20 steps
            base_f = self.f[l]
            # Increase ignition if time_factor is 1
            self.f[l] = base_f + 0.0005 * time_factor
            # Also add small random jitter
            self.f[l] += 0.0001 * (random.random() - 0.5)
            # ensure non-negative
            self.f[l] = max(0.0, self.f[l])

        # Prepare a new grid for updates
        new_grid = np.copy(self.grid)

        for l in range(self.layers):

            for r in range(self.height):
                for c in range(self.width):
                    cell_val = self.grid[l, r, c]

                    # 2) Wind weighting + dryness factor in neighbor summation
                    neighbor_burning_total = 0.0
                    total_weight = 0.0

                    for dl in [-1, 0, 1]:
                        ll = l + dl
                        if 0 <= ll < self.layers:
                            for dr in [-1, 0, 1]:
                                rr = (r + dr) % self.height
                                for dc in [-1, 0, 1]:
                                    cc = (c + dc) % self.width
                                    # skip the cell itself
                                    if dl == 0 and dr == 0 and dc == 0:
                                        continue
                                    # wind from the east => neighbors to the WEST have double weight
                                    weight = 1.0
                                    if dc < 0:  # west side neighbor
                                        weight = 2.0
                                    neighbor_burning_total += weight * self.grid[ll, rr, cc]
                                    total_weight += weight

                    # fraction of neighbors burning, factoring in the wind weighting
                    burning_fraction = neighbor_burning_total / total_weight if total_weight > 0 else 0.0

                    # dryness factor for column c => 0 at left, 1 at right
                    dryness_factor = c / float(self.width)
                    # scale local ignition & spread with dryness
                    local_f = self.f[l] * (1.0 + dryness_factor)
                    local_spread = self.spread_factor[l] * (1.0 + dryness_factor)

                    # 3) Burning logic
                    if cell_val >= 0.99:
                        # cell is burning => burn it down
                        new_val = cell_val - self.burn_rate[l]
                        if new_val < 0.0:
                            new_val = 0.0
                        new_grid[l, r, c] = new_val

                    else:
                        # Possibly spontaneously ignite
                        if cell_val > 0.1 and random.random() < local_f:
                            new_grid[l, r, c] = min(1.0, cell_val + 0.9)

                        # Possibly spontaneously grow if empty
                        elif cell_val < 0.1 and random.random() < self.p[l]:
                            new_grid[l, r, c] = 0.3 + 0.2 * random.random()

                        else:
                            # Fire spread from neighbors
                            spread_push = burning_fraction * local_spread
                            new_val = cell_val + spread_push
                            if new_val > 1.0:
                                new_val = 1.0
                            new_grid[l, r, c] = new_val

        # Replace old grid with new
        self.grid = new_grid
        self.step_count += 1  # track steps

    def inject_inputs(self, inputs):
        """
        If hooking up fuzzy neuron outputs or other external signals,
        interpret them as partial ignition or growth, etc.
        """
        for val in inputs:
            l = random.randint(0, self.layers - 1)
            r = random.randint(0, self.height - 1)
            c = random.randint(0, self.width - 1)

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
    Called each frame by FuncAnimation:
      1) update the CA
      2) update each layer's imshow
      3) update textual info
    """
    ca_obj.update()
    avg_val = ca_obj.get_average_value()

    for l, im in enumerate(images):
        im.set_array(ca_obj.grid[l])
    text.set_text(f"Step {frame_index+1}, Avg: {avg_val:.3f}")

    return images + [text]

if __name__ == "__main__":
    # Instantiate with 3 layers, each having default flavors
    ca = MultiLayerFuzzyForestFireCA(layers=3, width=50, height=50)

    # Set up matplotlib figure
    fig, axes = plt.subplots(1, ca.layers, figsize=(12, 4))
    images = []
    for l in range(ca.layers):
        ax = axes[l]
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
        fargs=(ca, images, text),
        frames=200,
        interval=200,
        blit=False
    )

    plt.tight_layout()
    plt.show()
