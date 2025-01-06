import numpy as np
import random

# -----------------------------------------------------
# 1. Fuzzy Logic Gates (same as before)
# -----------------------------------------------------
def fuzzy_and(a: float, b: float) -> float:
    return min(a, b)

def fuzzy_or(a: float, b: float) -> float:
    return max(a, b)

def fuzzy_not(a: float) -> float:
    return 1.0 - a

def fuzzy_nand(a: float, b: float) -> float:
    return 1.0 - fuzzy_and(a, b)

def fuzzy_nor(a: float, b: float) -> float:
    return 1.0 - fuzzy_or(a, b)

def fuzzy_xor(a: float, b: float) -> float:
    # (a OR b) - (a AND b)
    return fuzzy_or(a, b) - fuzzy_and(a, b)

# -----------------------------------------------------
# 2. Fuzzy Neuron / Neuron Layer (same as before)
# -----------------------------------------------------
class FuzzyNeuron:
    def __init__(self, gate_type='AND'):
        self.gate_type = gate_type

    def process(self, inputs):
        if len(inputs) == 0:
            return 0.0
        if len(inputs) == 1:
            if self.gate_type == 'NOT':
                return fuzzy_not(inputs[0])
            else:
                return inputs[0]

        # At least 2 inputs
        a = inputs[0]
        b = inputs[1]
        if self.gate_type == 'AND':
            out = fuzzy_and(a, b)
        elif self.gate_type == 'OR':
            out = fuzzy_or(a, b)
        elif self.gate_type == 'NAND':
            out = fuzzy_nand(a, b)
        elif self.gate_type == 'NOR':
            out = fuzzy_nor(a, b)
        elif self.gate_type == 'XOR':
            out = fuzzy_xor(a, b)
        else:
            out = fuzzy_and(a, b)  # fallback

        if len(inputs) == 3:
            c = inputs[2]
            if self.gate_type == 'AND':
                out = fuzzy_and(out, c)
            elif self.gate_type == 'OR':
                out = fuzzy_or(out, c)
            elif self.gate_type == 'NAND':
                out = fuzzy_nand(out, c)
            elif self.gate_type == 'NOR':
                out = fuzzy_nor(out, c)
            elif self.gate_type == 'XOR':
                out = fuzzy_xor(out, c)
            else:
                out = fuzzy_and(out, c)

        return out

class FuzzyNeuronLayer:
    def __init__(self, num_neurons=100, gate_types=None):
        self.num_neurons = num_neurons
        if gate_types is None:
            gate_types = ['AND', 'OR', 'XOR', 'NAND', 'NOR']
        self.neurons = []
        for _ in range(num_neurons):
            chosen_type = random.choice(gate_types)
            self.neurons.append(FuzzyNeuron(gate_type=chosen_type))

    def process_layer(self, input_values):
        outputs = []
        chunk_size = max(1, len(input_values) // self.num_neurons)
        for i, neuron in enumerate(self.neurons):
            start = i * chunk_size
            end = (i+1) * chunk_size
            slice_vals = input_values[start:end]
            if len(slice_vals) > 3:
                slice_vals = slice_vals[:3]
            if len(slice_vals) == 0:
                slice_vals = [0.0]
            outputs.append(neuron.process(slice_vals))
        return outputs

# -----------------------------------------------------
# 3. Multi-Layer Cellular Automaton
# -----------------------------------------------------
class MultiLayerCellularAutomaton:
    """
    Maintains multiple layers of 2D grids.
    Each layer can see neighbors in its own layer
    AND the corresponding cells (and neighbors) in
    adjacent layers above/below.
    """
    def __init__(self, layers=3, width=20, height=20):
        self.layers = layers
        self.width = width
        self.height = height
        # We'll store the CA as a 3D numpy array: shape (layers, height, width)
        # random float values in [0,1]
        self.grid = np.random.rand(layers, height, width)

    def update(self):
        """
        Each cell in each layer is updated based on:
         - Its own layer's neighbors (including itself)
         - Possibly the same coordinates in the layer above and below
         - We'll do a simple fuzzy-lifecycle rule again, just as a demonstration
        """
        new_grid = np.copy(self.grid)

        for l in range(self.layers):
            for r in range(self.height):
                for c in range(self.width):
                    # We'll gather the sum of neighbors in the same layer
                    total = 0.0
                    count = 0

                    # We'll do a 3D neighborhood that includes:
                    #   layer (l-1, l, l+1), row (r-1, r, r+1), col (c-1, c, c+1)
                    for dl in [-1, 0, 1]:
                        ll = l + dl
                        if 0 <= ll < self.layers:
                            for dr in [-1, 0, 1]:
                                rr = (r + dr) % self.height
                                for dc in [-1, 0, 1]:
                                    cc = (c + dc) % self.width
                                    total += self.grid[ll, rr, cc]
                                    count += 1

                    avg_val = total / count
                    # We'll do the same threshold logic:
                    if avg_val > 0.5:
                        new_grid[l, r, c] = min(avg_val + 0.1, 1.0)
                    else:
                        new_grid[l, r, c] = max(avg_val - 0.1, 0.0)

        self.grid = new_grid

    def inject_inputs(self, inputs):
        """
        We can choose to inject these fuzzy neuron outputs into random positions
        across *any* layer. For instance, if you have L layers, you might choose
        one random layer for each input, or distribute them evenly.
        """
        for val in inputs:
            # pick random layer
            l = random.randint(0, self.layers - 1)
            # pick random cell in that layer
            r = random.randint(0, self.height - 1)
            c = random.randint(0, self.width - 1)
            # combine fuzzily
            self.grid[l, r, c] = fuzzy_or(self.grid[l, r, c], val)

    def get_average_value(self):
        """
        Handy method to see how "lit" the entire multi-layer CA is.
        """
        return float(np.mean(self.grid))

# -----------------------------------------------------
# 4. Synthetic Ganglia Controller for Multi-Layer CA
# -----------------------------------------------------
class SyntheticGanglia:
    def __init__(self, neuron_layer: FuzzyNeuronLayer, ca: MultiLayerCellularAutomaton):
        self.neuron_layer = neuron_layer
        self.ca = ca

    def step(self, external_inputs: list):
        """
        1. Feed external inputs into the fuzzy neuron layer.
        2. Inject layer outputs into the multi-layer CA.
        3. Update CA.
        4. Return a simple metric (e.g., average cell value).
        """
        layer_outputs = self.neuron_layer.process_layer(external_inputs)
        self.ca.inject_inputs(layer_outputs)
        self.ca.update()
        return self.ca.get_average_value()

# -----------------------------------------------------
# 5. Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    # Create fuzzy neuron layer
    neuron_layer = FuzzyNeuronLayer(num_neurons=100)
    # Create multi-layer CA: e.g., 3 layers of 20x20
    ml_ca = MultiLayerCellularAutomaton(layers=3, width=20, height=20)
    # Create the synthetic ganglia
    ganglia = SyntheticGanglia(neuron_layer, ml_ca)

    # Run a few simulation steps
    for step_idx in range(10):
        # Provide some random external inputs
        external_inputs = [random.random() for _ in range(300)]
        avg_val = ganglia.step(external_inputs)
        print(f"Step {step_idx+1}, Average CA cell value: {avg_val:.3f}")
