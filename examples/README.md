# AgentEvolver Polymer Design Integration Examples

This directory contains example code and scaffolding for integrating [modelscope/AgentEvolver](https://github.com/modelscope/AgentEvolver) into polymer materials autonomous design workflows.

## 📁 Directory Structure

```
examples/
├── __init__.py              # Package initialization
├── README.md                # This file
├── gym_env.py               # OpenAI Gym environment wrapper
├── mutation_operators.py    # Domain-aware mutation operators
├── reward_function.py       # Multi-objective reward function
└── llm_planner.py          # LLM adapter for design planning
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install gymnasium numpy

# Optional: For LLM planner (if using OpenAI)
pip install openai

# Optional: For AgentEvolver integration
pip install modelscope-agent
```

### Running the Gym Environment Example

```bash
cd examples/
python gym_env.py
```

This will run a demo episode with a random agent exploring the polymer design space.

### Testing Mutation Operators

```bash
python mutation_operators.py
```

Demonstrates various mutation operators on a sample polymer structure.

### Testing Reward Function

```bash
python reward_function.py
```

Shows reward computation for different polymer candidates with varying properties.

### Testing LLM Planner

```bash
# Set API key (if using OpenAI)
export OPENAI_API_KEY="your-api-key"

python llm_planner.py
```

Generates design suggestions using LLM (or mock suggestions if no API key).

## 📚 Usage Examples

### Using the Gym Environment

```python
from examples import PolymerSurrogateEnv

# Create environment
env = PolymerSurrogateEnv(
    max_steps=50,
    target_properties={
        "tg": {"min": 150, "max": 200},
        "dielectric": {"min": 2.5, "max": 4.0}
    }
)

# Run episode
obs, info = env.reset()
for step in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Using Mutation Operators

```python
from examples import mutate_substitute, mutate_block_length

bigsmiles = "{[>][<]CC(C)(C)[>]}[<]"

# Substitute monomer
modified = mutate_substitute(bigsmiles, new_monomer="aromatic_A")

# Change block length
modified = mutate_block_length(bigsmiles, new_length=100)
```

### Computing Rewards

```python
from examples import compute_reward

properties = {
    "tg": 165.0,
    "dielectric": 3.5,
    "cost": 15.0
}

reward = compute_reward(
    props=properties,
    bigsmiles="{[>][<]c1ccccc1[>]}[<]",
    weights={"tg": 0.4, "dielectric": 0.3, "cost": 0.2, "feasibility": 0.1}
)

print(f"Reward: {reward:.3f}")
```

### Using LLM Planner

```python
from examples import LLMPlannerAdapter

planner = LLMPlannerAdapter(api_key="your-api-key")

plan = planner.generate_plan(
    current_bigsmiles="{[>][<]CC[>]}[<]",
    current_properties={"tg": 120, "dielectric": 4.0},
    target_properties={"tg": 160, "dielectric": 3.5}
)

for suggestion in plan["suggestions"]:
    print(f"Action: {suggestion['action']}")
    print(f"Details: {suggestion['details']}")
    print(f"Reasoning: {suggestion['reasoning']}")
```

## 🔌 Integration with AgentEvolver

### Basic Integration

```python
from examples import PolymerSurrogateEnv
from modelscope_agent import Agent  # Placeholder - adjust import based on actual SDK

# Create environment
env = PolymerSurrogateEnv()

# Create AgentEvolver agent (example - adjust based on actual API)
agent = Agent(
    env=env,
    algorithm="PPO",  # or "SAC", "DQN", etc.
    hyperparameters={
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64
    }
)

# Train agent
agent.train(total_timesteps=100000)

# Evaluate agent
obs, info = env.reset()
for step in range(50):
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### With LLM Guidance

```python
from examples import PolymerSurrogateEnv, LLMPlannerAdapter

env = PolymerSurrogateEnv()
planner = LLMPlannerAdapter()

# Use LLM to guide exploration
obs, info = env.reset()
for step in range(50):
    # Get LLM suggestions
    plan = planner.generate_plan(
        current_bigsmiles=info["polymer"],
        current_properties=info["properties"],
        target_properties={"tg": 160, "dielectric": 3.5}
    )
    
    # Use suggestions to bias action selection
    # (Implementation depends on how you want to integrate LLM guidance)
    action = select_action_from_plan(plan, env.action_space)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

## ⚙️ Customization

### Adding Custom Surrogate Models

Replace `DummySurrogatePredictor` in `gym_env.py`:

```python
from examples import PolymerSurrogateEnv

class MyCustomSurrogate:
    def predict(self, bigsmiles: str) -> dict:
        # Your trained model inference here
        encoded = encode_bigsmiles(bigsmiles)
        predictions = self.model(encoded)
        return {
            "tg": predictions[0],
            "dielectric": predictions[1],
            "cost": predictions[2],
            "uncertainty_tg": predictions[3],
            "uncertainty_dielectric": predictions[4]
        }

# Use custom surrogate
env = PolymerSurrogateEnv(surrogate_predictor=MyCustomSurrogate())
```

### Customizing Reward Function

Modify weights and targets:

```python
from examples import compute_reward

# Custom weights (prioritize Tg)
custom_weights = {
    "tg": 0.6,
    "dielectric": 0.2,
    "cost": 0.15,
    "feasibility": 0.05
}

# Custom target ranges
custom_targets = {
    "tg": {"min": 150, "max": 200, "ideal": 175},
    "dielectric": {"min": 2.0, "max": 4.5, "ideal": 3.0},
    "cost": {"min": 5.0, "max": 40.0, "ideal": 10.0}
}

reward = compute_reward(
    props=properties,
    bigsmiles=bigsmiles,
    weights=custom_weights,
    targets=custom_targets
)
```

### Adding New Mutation Operators

Extend `mutation_operators.py`:

```python
def mutate_crosslink_density(bigsmiles: str, density: float) -> str:
    """Add crosslinking annotations to the polymer."""
    # Your implementation here
    return modified_bigsmiles

# Register in combine_mutations
mutation_funcs["mutate_crosslink_density"] = mutate_crosslink_density
```

## 📋 TODO and Integration Checklist

Before using in production, replace placeholder implementations:

- [ ] **Surrogate Models**: Replace `DummySurrogatePredictor` with trained ML models
- [ ] **BigSMILES Parsing**: Integrate proper BigSMILES grammar parser and validator
- [ ] **Mutation Operators**: Implement domain-specific mutation logic with cheminformatics library (RDKit, OpenBabel)
- [ ] **Monomer Library**: Build database of monomers with properties and availability
- [ ] **LLM Integration**: Configure API keys and implement robust error handling
- [ ] **Observation Encoding**: Use learned embeddings (GNN, transformer) instead of placeholder encoding
- [ ] **Validation**: Add comprehensive unit tests for all components
- [ ] **Active Learning**: Implement acquisition functions and model update pipeline
- [ ] **Explainability**: Integrate SHAP, counterfactuals, and audit logging

## 🔒 Security Notes

- **API Keys**: Never commit API keys to version control. Use environment variables or secrets management.
- **Input Validation**: Always validate BigSMILES inputs to prevent injection attacks.
- **Model Security**: Protect trained surrogate models from unauthorized access.
- **Data Privacy**: Handle proprietary polymer structures and experimental data securely.

## 📖 Documentation

See the `docs/` directory for detailed documentation:

- [`docs/architecture.md`](../docs/architecture.md) - System architecture overview
- [`docs/integration_plan.md`](../docs/integration_plan.md) - Step-by-step integration guide
- [`docs/explainability.md`](../docs/explainability.md) - Explainability techniques and audit trails

## 📝 License

MIT License - See repository root for details

## 🤝 Contributing

This is scaffolding code intended to be customized for your specific polymer design workflow. Contributions and improvements are welcome!

## 📧 Contact

For questions about AgentEvolver integration, refer to:
- [AgentEvolver GitHub](https://github.com/modelscope/AgentEvolver)
- [BigSMILES Documentation](https://olsenlabmit.github.io/BigSMILES/)
- [OpenAI Gym Documentation](https://gymnasium.farama.org/)
