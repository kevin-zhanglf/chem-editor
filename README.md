# chem-editor

A repository for polymer materials design tools and AgentEvolver integration examples.

## 🌟 Features

- **AgentEvolver Integration**: Scaffolding for integrating AI-driven polymer design with modelscope/AgentEvolver
- **OpenAI Gym Environment**: Custom environment for reinforcement learning in polymer design space
- **Surrogate Model Interface**: Wrapper for property prediction models
- **Domain-Aware Mutations**: Chemical structure modification operators
- **Multi-Objective Rewards**: Configurable reward functions for property optimization
- **LLM Planning**: Adapter for large language model-based design suggestions

## 📁 Repository Structure

```
chem-editor/
├── docs/                    # Documentation
│   ├── architecture.md      # System architecture overview
│   ├── integration_plan.md  # Step-by-step integration guide
│   └── explainability.md    # Explainability techniques
├── examples/                # Integration examples and scaffolding
│   ├── gym_env.py          # OpenAI Gym environment
│   ├── mutation_operators.py # Polymer mutation operators
│   ├── reward_function.py  # Multi-objective reward computation
│   ├── llm_planner.py      # LLM adapter for planning
│   └── README.md           # Examples documentation
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kevin-zhanglf/chem-editor.git
cd chem-editor

# Install dependencies
pip install gymnasium numpy

# Optional: For LLM features
pip install openai

# Optional: For AgentEvolver
pip install modelscope-agent
```

### Running Examples

```bash
# Test the Gym environment
cd examples
python gym_env.py

# Test mutation operators
python mutation_operators.py

# Test reward function
python reward_function.py

# Test LLM planner (requires API key)
export OPENAI_API_KEY="your-key"
python llm_planner.py
```

## 📚 Documentation

- **[Architecture](docs/architecture.md)**: System design for AgentEvolver integration
- **[Integration Plan](docs/integration_plan.md)**: 5-phase implementation roadmap
- **[Explainability](docs/explainability.md)**: Techniques for interpretable AI-driven design
- **[Examples README](examples/README.md)**: Detailed usage guide for example code

## 🔧 Integration with AgentEvolver

This repository provides scaffolding to integrate [modelscope/AgentEvolver](https://github.com/modelscope/AgentEvolver) into polymer materials design workflows:

1. **Environment Layer**: OpenAI Gym wrapper for surrogate models
2. **Agent Layer**: RL/LLM agents explore the design space
3. **Active Learning**: Efficient experimental validation
4. **Explainability**: Interpret agent decisions and model predictions

See [`examples/README.md`](examples/README.md) for detailed integration instructions.

## ⚠️ Important Notes

This code is **scaffolding and examples** intended for customization:

- Replace placeholder surrogate models with your trained models
- Implement domain-specific BigSMILES parsing and validation
- Add real monomer libraries and chemical feasibility checks
- Configure LLM API keys securely (environment variables)
- Extend mutation operators with cheminformatics libraries (RDKit, etc.)

See TODO markers in code for areas requiring domain-specific implementation.

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 🔗 References

- [AgentEvolver GitHub](https://github.com/modelscope/AgentEvolver)
- [BigSMILES Specification](https://doi.org/10.1021/acscentsci.9b00476)
- [OpenAI Gym Documentation](https://gymnasium.farama.org/)
- [ModelScope](https://www.modelscope.cn/)