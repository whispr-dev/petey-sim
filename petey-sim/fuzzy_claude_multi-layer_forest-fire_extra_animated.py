import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import List, Optional, Dict
import json

@dataclass
class LayerConfig:
    """Configuration parameters for a single CA layer"""
    growth_rate: float           # p: probability of new growth
    ignition_rate: float        # f: spontaneous ignition
    burn_rate: float           # rate of burning down
    spread_factor: float       # rate of fire spread
    attention_weight: float = 1.0  # weight for LLM attention

class MultiLayerFuzzyForestFireCA:
    """
    Enhanced multi-layer fuzzy forest-fire CA with LLM integration capabilities.
    Includes:
    - Structured input/output layers
    - Attention-weighted state updates
    - Hierarchical processing
    - State persistence
    - Configuration management
    """

    def __init__(self, 
                 layers: int = 3,
                 width: int = 40, 
                 height: int = 40,
                 configs: Optional[List[LayerConfig]] = None,
                 hierarchy_factor: float = 0.5):
        """
        Initialize the CA with enhanced configuration options.
        
        Args:
            layers: Number of CA layers
            width: Grid width
            height: Grid height
            configs: List of LayerConfig objects for each layer
            hierarchy_factor: Strength of hierarchical coupling (0-1)
        """
        self.layers = layers
        self.width = width
        self.height = height
        self.hierarchy_factor = hierarchy_factor
        
        # Initialize configs with defaults if none provided
        if configs is None:
            configs = [
                LayerConfig(0.02, 0.002, 0.6, 0.45, 0.8),  # Top layer - broad patterns
                LayerConfig(0.02, 0.0004, 0.4, 0.36, 0.9),  # Middle layer
                LayerConfig(0.01, 0.0003, 0.1, 0.17, 0.25)   # Bottom layer - fine details
            ]
        
        self.configs = configs
        
        # Convert configs to numpy arrays for efficient processing
        self.p = np.array([c.growth_rate for c in configs])
        self.f = np.array([c.ignition_rate for c in configs])
        self.burn_rate = np.array([c.burn_rate for c in configs])
        self.spread_factor = np.array([c.spread_factor for c in configs])
        self.attention_weights = np.array([c.attention_weight for c in configs])
        
        # Initialize grid with random values
        self.grid = np.random.rand(layers, height, width)
        
        # 2% initial burning cells
        burning_mask = (np.random.rand(layers, height, width) < 0.02)
        self.grid[burning_mask] = 1.0
        
        # Track simulation steps and state history
        self.step_count = 0
        self.state_history = []
        
        # Initialize layer-specific update matrices
        self.layer_coupling = self._initialize_layer_coupling()

    def _initialize_layer_coupling(self) -> np.ndarray:
        """
        Initialize the hierarchical coupling matrix between layers.
        Higher layers influence lower layers more than vice versa.
        """
        coupling = np.zeros((self.layers, self.layers))
        for i in range(self.layers):
            for j in range(self.layers):
                if i == j:
                    coupling[i,j] = 1.0
                elif i < j:  # higher layer to lower
                    coupling[i,j] = self.hierarchy_factor * (1.0 - abs(i-j)/self.layers)
                else:  # lower layer to higher
                    coupling[i,j] = (self.hierarchy_factor * 0.5) * (1.0 - abs(i-j)/self.layers)
        return coupling

    def update(self):
        """
        Enhanced update method with hierarchical processing and attention-weighted updates.
        """
        # Time-based parameter changes
        self._update_parameters()
        
        # Prepare new grid
        new_grid = np.copy(self.grid)
        
        # Calculate global state influence
        global_state = np.sum(self.grid * self.attention_weights[:, np.newaxis, np.newaxis], axis=0)
        global_state /= np.sum(self.attention_weights)
        
        # Update each layer
        for l in range(self.layers):
            # Calculate hierarchical influence
            hierarchy_influence = np.zeros_like(self.grid[0])
            for other_l in range(self.layers):
                if other_l != l:
                    hierarchy_influence += self.layer_coupling[l,other_l] * self.grid[other_l]
            
            new_grid[l] = self._update_layer(l, hierarchy_influence, global_state)
        
        # Store state snapshot every 10 steps
        if self.step_count % 10 == 0:
            self.state_history.append(self.get_state_snapshot())
        
        # Update grid and step count
        self.grid = new_grid
        self.step_count += 1

    def _update_layer(self, layer: int, hierarchy_influence: np.ndarray, 
                     global_state: np.ndarray) -> np.ndarray:
        """
        Update a single layer with hierarchical influence and global state.
        """
        new_layer = np.copy(self.grid[layer])
        
        # Vectorized neighbor calculation
        padding = np.pad(self.grid[layer], ((1,1), (1,1)), mode='wrap')
        
        # Calculate weighted neighbor sums with wind effect
        neighbor_sums = np.zeros_like(self.grid[layer])
        wind_weights = np.array([[0.4, 0.7, 2.0],  # West neighbors weighted more
                               [0.4, 0.1, 1.3],
                               [0.2, 1.3, 1.9]])
        
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                neighbor_sums += (wind_weights[i,j] * 
                                np.roll(np.roll(padding[1:-1,1:-1], i-1, 0), j-1, 1))
        
        # Column-based dryness factor
        dryness = np.linspace(0, 1, self.width)
        dryness = np.tile(dryness, (self.height, 1))
        
        # Combined influence factors
        influence = (0.5 * neighbor_sums + 
                    0.2 * hierarchy_influence +
                    0.1 * global_state)
        
        # State updates with fuzzy rules
        burning_mask = self.grid[layer] >= 0.99
        empty_mask = self.grid[layer] < 0.1
        tree_mask = ~(burning_mask | empty_mask)
        
        # Update burning cells
        new_layer[burning_mask] -= self.burn_rate[layer]
        
        # Update empty cells
        growth_prob = self.p[layer] * (1 + 0.2 * influence)
        growth_mask = empty_mask & (np.random.rand(*self.grid[layer].shape) < growth_prob)
        new_layer[growth_mask] = 0.3 + 0.2 * np.random.rand(growth_mask.sum())
        
        # Update trees
        ignition_prob = self.f[layer] * (1 + dryness) * (1 + 0.3 * influence)
        ignition_mask = tree_mask & (np.random.rand(*self.grid[layer].shape) < ignition_prob)
        new_layer[ignition_mask] = 1.0
        
        # Apply spread
        spread = self.spread_factor[layer] * influence * (1 + dryness)
        new_layer[tree_mask] += spread[tree_mask]
        
        # Ensure bounds
        return np.clip(new_layer, 0, 1)

    def inject_llm_input(self, token_activations: Dict[int, float]):
        """
        Inject LLM token activations into the CA state.
        
        Args:
            token_activations: Dict mapping token IDs to activation values [0,1]
        """
        # Normalize activations
        values = np.array(list(token_activations.values()))
        normalized = (values - values.min()) / (values.max() - values.min())
        
        # Map tokens to grid positions using a simple hash
        for token_id, activation in zip(token_activations.keys(), normalized):
            l = token_id % self.layers
            r = (token_id // self.layers) % self.height
            c = (token_id // (self.layers * self.height)) % self.width
            
            # Weight injection by layer attention
            weighted_activation = activation * self.attention_weights[l]
            
            # Inject as partial activation
            self.grid[l, r, c] = min(1.0, 
                                   self.grid[l, r, c] + 0.3 * weighted_activation)

    def get_llm_output(self, num_tokens: int) -> Dict[int, float]:
        """
        Extract token activations from CA state for LLM consumption.
        
        Args:
            num_tokens: Number of token activations to extract
            
        Returns:
            Dict mapping token IDs to activation values
        """
        token_activations = {}
        
        # Sample grid points weighted by attention
        for token_id in range(num_tokens):
            l = token_id % self.layers
            r = (token_id // self.layers) % self.height
            c = (token_id // (self.layers * self.height)) % self.width
            
            # Weight activation by layer attention
            activation = self.grid[l, r, c] * self.attention_weights[l]
            token_activations[token_id] = float(activation)
            
        return token_activations

    def get_state_snapshot(self) -> Dict:
        """Get a serializable snapshot of the current CA state."""
        return {
            'step': self.step_count,
            'grid': self.grid.tolist(),
            'params': {
                'p': self.p.tolist(),
                'f': self.f.tolist(),
                'burn_rate': self.burn_rate.tolist(),
                'spread_factor': self.spread_factor.tolist()
            }
        }

    def save_state(self, filepath: str):
        """Save the current state to a JSON file."""
        state = self.get_state_snapshot()
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load_state(self, filepath: str):
        """Load state from a JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.step_count = state['step']
        self.grid = np.array(state['grid'])
        self.p = np.array(state['params']['p'])
        self.f = np.array(state['params']['f'])
        self.burn_rate = np.array(state['params']['burn_rate'])
        self.spread_factor = np.array(state['params']['spread_factor'])

    def _update_parameters(self):
        """Update time-varying parameters."""
        time_factor = (self.step_count // 20) % 2
        
        for l in range(self.layers):
            base_f = self.configs[l].ignition_rate
            self.f[l] = base_f + 0.0005 * time_factor + 0.0003 * (random.random() - 0.5)
            self.f[l] = max(0.0, self.f[l])

def visualize_ca(ca: MultiLayerFuzzyForestFireCA, num_frames: int = 200):
    """Enhanced visualization with attention weights and hierarchical influence."""
    fig, axes = plt.subplots(1, ca.layers, figsize=(15, 5))
    images = []
    
    for l in range(ca.layers):
        ax = axes[l]
        im = ax.imshow(ca.grid[l], cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f"Layer {l}\nAttention: {ca.attention_weights[l]:.2f}")
        ax.axis('off')
        images.append(im)
    
    text = axes[0].text(0.02, 1.05, "", transform=axes[0].transAxes, color="white")
    
    def update_frame(frame_index):
        ca.update()
        avg_val = np.mean(ca.grid)
        
        for l, im in enumerate(images):
            im.set_array(ca.grid[l])
        text.set_text(f"Step {frame_index+1}, Avg: {avg_val:.3f}")
        
        return images + [text]
    
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=num_frames,
        interval=200,
        blit=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage with new features
    
    # Create custom layer configs
    configs = [
        LayerConfig(0.01, 0.001, 0.5, 0.6, 1.0),  # Top layer
        LayerConfig(0.03, 0.0005, 0.3, 0.35, 0.7),  # Middle layer
        LayerConfig(0.02, 0.0001, 0.2, 0.15, 0.4)   # Bottom layer
    ]
    
    # Initialize CA with configs and hierarchical coupling
    ca = MultiLayerFuzzyForestFireCA(
        layers=3,
        width=70,
        height=70,
        configs=configs,
        hierarchy_factor=0.5
    )
    
    # Example LLM integration
    token_activations = {0: 0.8, 1: 0.5, 2: 0.3}  # Example token activations
    ca.inject_llm_input(token_activations)
    
    # Run visualization
    visualize_ca(ca, num_frames=200)
