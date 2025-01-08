import numpy as np
import random

# -----------------------------------------------------
# 1. Fuzzy Logic Gates
# -----------------------------------------------------
def fuzzy_and(a: float, b: float) -> float:
    """
    Returns a value representing the AND of two fuzzy inputs.
    Range of a, b: 0.0 to 1.0
    """
    return min(a, b)

def fuzzy_or(a: float, b: float) -> float:
    """
    Returns a value representing the OR of two fuzzy inputs.
    """
    return max(a, b)

def fuzzy_not(a: float) -> float:
    """
    Returns a value representing the NOT of a fuzzy input.
    """
    return 1.0 - a

def fuzzy_nand(a: float, b: float) -> float:
    """
    Returns a value representing the NAND of two fuzzy inputs.
    """
    return 1.0 - fuzzy_and(a, b)

def fuzzy_nor(a: float, b: float) -> float:
    """
    Returns a value representing the NOR of two fuzzy inputs.
    """
    return 1.0 - fuzzy_or(a, b)

def fuzzy_xor(a: float, b: float) -> float:
    """
    Returns a value representing the XOR of two fuzzy inputs.
    For fuzzy logic, this is one simple approximation.
    """
    # In fuzzy logic, XOR can be taken as (a OR b) - (a AND b)
    return fuzzy_or(a, b) - fuzzy_and(a, b)

# -----------------------------------------------------
# 2. Fuzzy Neuron Layer
#    Let's define a basic neuron that can handle up to
#    three fuzzy inputs and combine them in a gate.
# -----------------------------------------------------
class FuzzyNeuron:
    def __init__(self, gate_type='AND'):
        """
        gate_type can be 'AND', 'OR', 'NAND', 'NOR', 'XOR', etc.
        """
        self.gate_type = gate_type

    def process(self, inputs):
        """
        Accepts a list of up to 3 inputs (floats),
        returns a single float output.
        """
        if len(inputs) == 0:
            return 0.0
        if len(inputs) == 1:
            # If only one input, let's just pass it or do fuzzy_not
            if self.gate_type == 'NOT':
                return fuzzy_not(inputs[0])
            else:
                # By default, pass-through if there's no second input
                return inputs[0]

        # For 2 or more inputs:
        a = inputs[0]
        b = inputs[1]

        # Combine first two
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
            out = fuzzy_and(a, b)  # Default fallback

        # If there's a third input, combine again
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
        self.neurons = []
        if gate_types is None:
            gate_types = ['AND', 'OR', 'XOR', 'NAND', 'NOR']
        # Randomly assign gate types or pick from the list
        for _ in range(num_neurons):
            chosen_type = random.choice(gate_types)
            self.neurons.append(FuzzyNeuron(gate_type=chosen_type))

    def process_layer(self, input_values):
        """
        input_values: list of float values
        We'll chunk them into sets of 3 for each neuron (where possible).
        Return the outputs as a list of floats.
        """
        outputs = []
        chunk_size = max(1, len(input_values) // self.num_neurons)  # try to distribute inputs
        for i, neuron in enumerate(self.neurons):
            # Grab the relevant slice of input
            start = i * chunk_size
            end = (i+1) * chunk_size
            slice_vals = input_values[start:end]
            if len(slice_vals) > 3:
                slice_vals = slice_vals[:3]  # limit to 3 for simplicity
            # If no slice available, just feed 0
            if len(slice_vals) == 0:
                slice_vals = [0.0]
            outputs.append(neuron.process(slice_vals))
        return outputs

# -----------------------------------------------------
# 3. Cellular Automaton (CA) Interface
# -----------------------------------------------------
class CellularAutomaton:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        # Initialize a random grid of floats in [0, 1]
        self.grid = np.random.rand(self.height, self.width)

    def update(self):
        """
        Very simple CA rule (like a fuzzy conway’s game of life).
        We’ll average neighbors and apply a threshold.
        """
        new_grid = np.copy(self.grid)
        for r in range(self.height):
            for c in range(self.width):
                # Moore neighborhood
                total = 0.0
                count = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = (r + dr) % self.height
                        cc = (c + dc) % self.width
                        total += self.grid[rr, cc]
                        count += 1
                avg_val = total / count
                # Let’s impose a fuzzy threshold:
                if avg_val > 0.5:
                    new_grid[r, c] = min(avg_val + 0.1, 1.0)
                else:
                    new_grid[r, c] = max(avg_val - 0.1, 0.0)
        self.grid = new_grid

    def inject_inputs(self, inputs):
        """
        Possibly inject some fuzzy neuron outputs into random cells.
        """
        for val in inputs:
            r = random.randint(0, self.height - 1)
            c = random.randint(0, self.width - 1)
            # Combine with existing cell fuzzily
            self.grid[r, c] = fuzzy_or(self.grid[r, c], val)  # or pick any gate you like

# -----------------------------------------------------
# 4. Synthetic Ganglia Controller
#    Orchestrates data flow between FuzzyNeuronLayer and CA.
# -----------------------------------------------------
class SyntheticGanglia:
    def __init__(self, neuron_layer: FuzzyNeuronLayer, ca: CellularAutomaton):
        self.neuron_layer = neuron_layer
        self.ca = ca

    def step(self, external_inputs: list):
        """
        1. Send external inputs to the neuron layer.
        2. Take the neuron layer’s outputs, inject them into the CA.
        3. Update the CA.
        4. Possibly read back from the CA to feed into next step.
        """
        layer_outputs = self.neuron_layer.process_layer(external_inputs)
        self.ca.inject_inputs(layer_outputs)
        self.ca.update()

        # We could feed back a summary of the CA as input for the next round
        # For now, let's just gather the average cell value:
        average_ca_value = np.mean(self.ca.grid)
        return average_ca_value

# -----------------------------------------------------
# Example usage / demonstration
# -----------------------------------------------------
if __name__ == "__main__":
    # Create a fuzzy neuron layer with 100 neurons
    neuron_layer = FuzzyNeuronLayer(num_neurons=100)

    # Create a 20x20 CA
    ca = CellularAutomaton(width=20, height=20)

    # Create the synthetic ganglia
    ganglia = SyntheticGanglia(neuron_layer, ca)

    # Run a few simulation steps
    for step_idx in range(10):
        # Provide some random external inputs: e.g., 300 fuzzy values
        external_inputs = [random.random() for _ in range(300)]
        avg_val = ganglia.step(external_inputs)
        print(f"Step {step_idx+1}, Average CA cell value: {avg_val:.3f}")
