import numpy as np
import random
import matplotlib.pyplot as plt

class FuzzyForestFireCA:
    """
    Multi-layer fuzzy forest-fire CA.
    Each cell is a float in [0,1]:
      - ~0.0 => empty
      - (0,1) => tree-ish
      - 1.0   => burning
    We'll do very approximate transitions to illustrate
    how you might adapt the forest-fire concept fuzzily.
    """

    def __init__(self, layers=3, width=20, height=20,
                 p=0.01,    # Probability of spontaneous growth in an empty cell
                 f=0.0005,  # Probability of spontaneous ignition in a tree cell
                 burn_rate=0.5, # How fast burning cells go down
                 spread_factor=0.3 # How strongly burning neighbors push a cell towards ignition
                 ):
        """
        p => probability that an empty cell spontaneously grows some tree.
        f => probability that a tree spontaneously ignites.
        burn_rate => fraction of 'fire' lost each step if burning.
        spread_factor => how strongly neighbors that are burning push the cell to ignite
        """
        self.layers = layers
        self.width = width
        self.height = height
        self.p = p
        self.f = f
        self.burn_rate = burn_rate
        self.spread_factor = spread_factor

        # random init: ~ half-empty, half moderately treed
        # we'll keep some random fraction near 1.0 to represent burning
        self.grid = np.random.rand(layers, height, width)
        # Optionally, clamp some portion to near 1.0:
        burning_mask = (np.random.rand(layers, height, width) < 0.05)
        self.grid[burning_mask] = 1.0  # 5% random burning

    def update(self):
        new_grid = np.copy(self.grid)

        for l in range(self.layers):
            for r in range(self.height):
                for c in range(self.width):

                    cell_val = self.grid[l, r, c]
                    # We'll look at neighbors in this layer and above/below
                    neighbor_burning = 0.0

                    # 3D neighborhood
                    for dl in [-1, 0, 1]:
                        ll = l + dl
                        if 0 <= ll < self.layers:
                            for dr in [-1, 0, 1]:
                                rr = (r + dr) % self.height
                                for dc in [-1, 0, 1]:
                                    cc = (c + dc) % self.width
                                    if not (dl == 0 and dr == 0 and dc == 0):
                                        # Add up any neighbors that are "burning"
                                        neighbor_burning += self.grid[ll, rr, cc]

                    # If the cell is burning (>= ~1.0), burn it down
                    if cell_val >= 0.99:
                        # It's fully on fire, so let's reduce it
                        new_val = cell_val - self.burn_rate
                        if new_val < 0.0:
                            new_val = 0.0  # once fully burned, becomes empty
                        new_grid[l, r, c] = new_val

                    else:
                        # Not fully burning
                        # 1) Possibly spontaneously ignite
                        if cell_val > 0.1 and random.random() < self.f:
                            # jump cell value up to near 1.0 (burning)
                            new_grid[l, r, c] = min(1.0, cell_val + 0.9)
                        # 2) Possibly spontaneously grow if empty
                        elif cell_val < 0.1 and random.random() < self.p:
                            # jump up to a moderate tree value
                            new_grid[l, r, c] = 0.3 + 0.2 * random.random()
                        else:
                            # 3) Fire spread from neighbors
                            # We'll figure out how many neighbors are "really" burning
                            # neighbor_burning is a sum of up to 26 cells in 3D. If we want
                            # a fraction, we can scale it by the maximum possible sum (26).
                            # We'll do something simpler here, just see if it's "significant."
                            # The bigger neighbor_burning is, the more we push this cell up.
                            # This is fuzzy, so the push might be partial.
                            burn_push = (neighbor_burning / 26.0) * self.spread_factor
                            new_val = cell_val + burn_push
                            if new_val > 1.0:
                                new_val = 1.0
                            new_grid[l, r, c] = new_val

        self.grid = new_grid

    def get_average_value(self):
        return float(np.mean(self.grid))

    def inject_inputs(self, inputs):
        """
        If we want to incorporate your fuzzy neuron outputs, we can interpret them
        as either:
         - additional 'burn push' or
         - changes in tree density
        For now, let's just treat them as mild ignition triggers if they're > 0.7,
        or tree growth if they're ~0.3-0.7, etc.
        """
        for val in inputs:
            l = random.randint(0, self.layers - 1)
            r = random.randint(0, self.height - 1)
            c = random.randint(0, self.width - 1)

            cell_val = self.grid[l, r, c]
            if val > 0.7:
                # push it toward burning
                self.grid[l, r, c] = min(1.0, cell_val + 0.5)
            elif val > 0.3:
                # push it toward moderate tree
                self.grid[l, r, c] = max(cell_val, 0.5)
            else:
                # push it toward emptiness
                self.grid[l, r, c] = max(0.0, cell_val - 0.2)

# -----------------------------------------------------
# Demo / Usage
# -----------------------------------------------------
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Create an instance of the fuzzy forest fire CA
    ff_ca = FuzzyForestFireCA(
        layers=3,   # more layers = more "vertical" structure
        width=50,   # bigger width/height might help patterns
        height=50,
        p=0.02,     # up the growth a bit
        f=0.0005,   # chance of spontaneous ignition
        burn_rate=0.2,
        spread_factor=0.4
    )

    steps = 50

    for step_idx in range(steps):
        # You can provide some random or structured "fuzzy neuron outputs" if you want:
        # external_inputs = [random.random() for _ in range(20)]
        # ff_ca.inject_inputs(external_inputs)

        ff_ca.update()
        avg_val = ff_ca.get_average_value()
        print(f"Step {step_idx+1}, Average CA cell value: {avg_val:.3f}")

        # Plot each layer after each step
        fig, axes = plt.subplots(1, ff_ca.layers, figsize=(12, 4))
        for l in range(ff_ca.layers):
            ax = axes[l]
            im = ax.imshow(ff_ca.grid[l], cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f'Layer {l}, Step {step_idx+1}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
