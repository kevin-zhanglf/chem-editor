# System Architecture: AgentEvolver Integration for Polymer Materials Design

## Overview

This document describes the system architecture for integrating [modelscope/AgentEvolver](https://github.com/modelscope/AgentEvolver) into an autonomous polymer materials design workflow. The integration enables AI-driven exploration, optimization, and explainability in the polymer chemistry domain by combining reinforcement learning agents with surrogate models, active learning, and safety guardrails.

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                         User / Materials Scientist                     │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          Agent Layer (AgentEvolver)                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  LLM Planner     │  │  RL Policy       │  │  Mutation Engine   │  │
│  │  (Suggestions)   │  │  (Exploration)   │  │  (Operators)       │  │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘  │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    Environment Adapter (Gym Interface)                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  • action_space: Discrete/Continuous mutation actions            │ │
│  │  • observation_space: BigSMILES representation + property vector │ │
│  │  • step(): Apply mutation, query surrogate, compute reward       │ │
│  │  • reset(): Initialize polymer state                             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        Surrogate Models Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ Tg Predictor │  │ Dielectric   │  │ Cost Model   │  │ Other    │  │
│  │ (ML Model)   │  │ Predictor    │  │              │  │ Props    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘  │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│                            Data Layer                                  │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  • Polymer database (BigSMILES strings)                          │ │
│  │  • Historical experimental data                                  │ │
│  │  • Validation dataset for surrogate models                       │ │
│  │  • Exploration trajectory logs                                   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Data Layer

**Purpose**: Provides storage and retrieval for polymer structures, experimental data, and training datasets.

**Components**:
- **Polymer Database**: BigSMILES representations of known and candidate polymers
- **Experimental Data**: Ground-truth measurements (Tg, dielectric constant, mechanical properties, etc.)
- **Surrogate Training Data**: Preprocessed datasets for training and updating surrogate models
- **Exploration Logs**: Agent trajectories, rewards, and decisions for analysis and retraining

**Interfaces**:
- Query polymers by property ranges
- Store new experimental results
- Export datasets for model retraining

### 2. Surrogate Models Layer

**Purpose**: Fast prediction of polymer properties without running expensive simulations or experiments.

**Components**:
- **Property Predictors**: Machine learning models (e.g., graph neural networks, transformers) that map BigSMILES → property vectors
  - Glass transition temperature (Tg)
  - Dielectric constant
  - Mechanical properties
  - Solubility, cost estimates, etc.
- **Uncertainty Quantification**: Bayesian models or ensemble methods to estimate prediction confidence
- **Model Update Pipeline**: Active learning loop to retrain models with new experimental data

**Interfaces**:
- `predict(bigsmiles: str) → dict`: Returns property predictions and uncertainties
- `update(new_data: list) → None`: Retrains models with fresh experimental results

**Notes**:
- Surrogate models should be validated against experimental data
- Use cross-validation and hold-out test sets to assess generalization

### 3. Environment Adapter (Gym Interface)

**Purpose**: Wraps the polymer design problem as an OpenAI Gym-compatible environment for agent training.

**Components**:
- **State Representation**: Current polymer BigSMILES string + predicted property vector
- **Action Space**: Discrete or continuous actions corresponding to mutation operators
  - Substitute monomers
  - Change block lengths
  - Adjust composition ratios
  - Modify processing conditions (if applicable)
- **Reward Function**: Multi-objective weighted reward based on target properties and constraints
- **Transition Dynamics**: Applies mutation operators and queries surrogate models

**Key Methods**:
- `reset() → observation`: Initializes a random or specified polymer
- `step(action) → (observation, reward, done, info)`: Applies action, updates state, computes reward
- `render(mode='human')`: Visualizes current polymer and properties

**See**: `examples/gym_env.py` for implementation

### 4. Agent Layer (AgentEvolver)

**Purpose**: Leverages AgentEvolver's LLM-based and RL-based agents to explore the polymer design space.

**Components**:
- **LLM Planner**: Uses large language models to suggest high-level design strategies
  - Input: Current polymer, target properties, context
  - Output: Plan suggestions (e.g., "Increase Tg by adding aromatic groups")
- **RL Policy**: Reinforcement learning agent (PPO, SAC, etc.) trained to maximize cumulative reward
  - Learns mutation strategies through trial-and-error
  - Balances exploration (novel polymers) and exploitation (known good regions)
- **Mutation Engine**: Applies domain-aware transformations to polymer structures
  - Ensures chemical validity (via BigSMILES grammar)
  - Enforces constraints (e.g., commercial availability, cost limits)

**Integration Points**:
- AgentEvolver's `Agent` class wraps the Gym environment
- Custom reward shaping and curriculum learning can be implemented
- Multi-agent or hierarchical agent architectures are possible

**See**: `examples/llm_planner.py` and `examples/mutation_operators.py`

### 5. Active Learning and Explainability

**Purpose**: Efficiently selects polymers for experimental validation and provides interpretable insights.

**Active Learning**:
- **Acquisition Functions**: Select candidate polymers with high predicted performance and high uncertainty
- **Batch Selection**: Choose diverse candidates to maximize information gain per experiment
- **Model Update**: Incorporate experimental results to improve surrogate models

**Explainability**:
- **SHAP Values**: Identify which structural features contribute most to predicted properties
- **Counterfactual Analysis**: "If we changed monomer X to Y, Tg would increase by Z"
- **Rule Extraction**: Generate human-readable design rules from agent trajectories
- **Audit Trail**: Log all agent decisions, surrogate predictions, and experimental validations

**See**: `docs/explainability.md` for detailed techniques

### 6. Safety and Constraints

**Purpose**: Ensure agent proposals are chemically feasible, safe, and aligned with design goals.

**Safety Mechanisms**:
- **Validity Checks**: Reject polymers that violate BigSMILES grammar or chemical rules
- **Feasibility Filters**: Exclude structures with unavailable monomers or prohibitive costs
- **Property Bounds**: Enforce hard constraints (e.g., Tg must be between -50°C and 300°C)
- **Toxicity Screening**: Flag potentially hazardous compounds using lookup tables or models

**Implementation**:
- Add penalty terms to the reward function for constraint violations
- Use rule-based filters in the environment's `step()` method
- Incorporate human-in-the-loop review for high-risk candidates

### 7. Metrics and Evaluation

**Purpose**: Quantify agent performance and guide optimization.

**Key Metrics**:
- **Cumulative Reward**: Total reward over an episode (measures goal achievement)
- **Success Rate**: Fraction of episodes where target properties are met
- **Sample Efficiency**: Number of experimental validations needed to find optimal polymers
- **Diversity Score**: Structural diversity of explored candidates (avoid local optima)
- **Surrogate Accuracy**: MAE/RMSE between predicted and experimental properties

**Logging**:
- Use TensorBoard or Weights & Biases for real-time monitoring
- Store trajectories and hyperparameters for reproducibility

## Data Flow Example

1. **Initialization**: User specifies target properties (e.g., Tg > 150°C, low cost)
2. **Agent Start**: AgentEvolver agent resets the Gym environment, obtaining an initial polymer state
3. **Planning**: LLM planner suggests a high-level strategy (e.g., "Add aromatic monomers")
4. **Action Selection**: RL policy chooses a specific mutation action
5. **Environment Step**: 
   - Apply mutation operator to BigSMILES string
   - Query surrogate models for property predictions
   - Compute multi-objective reward
   - Check validity and safety constraints
6. **Feedback**: Agent receives new state and reward, updates policy
7. **Iteration**: Steps 3-6 repeat until episode terminates (max steps or goal reached)
8. **Active Learning**: Top candidates with high reward and high uncertainty are selected for experimental validation
9. **Model Update**: Experimental results are added to the data layer, and surrogate models are retrained
10. **Next Cycle**: Agent continues exploration with improved surrogate models

## Deployment Considerations

- **Scalability**: Run multiple agent instances in parallel to explore diverse regions of design space
- **Fault Tolerance**: Handle surrogate model failures gracefully (e.g., fallback to heuristics)
- **Versioning**: Track versions of surrogate models, agent policies, and datasets
- **Security**: Protect API keys for LLM services and restrict access to experimental data
- **Compliance**: Ensure data privacy and adhere to materials safety regulations

## References

- [AgentEvolver GitHub](https://github.com/modelscope/AgentEvolver)
- [OpenAI Gym Documentation](https://gymnasium.farama.org/)
- [BigSMILES Specification](https://doi.org/10.1021/acscentsci.9b00476)
- [Active Learning for Materials Discovery](https://doi.org/10.1038/s41524-020-00479-8)

## Next Steps

Refer to `docs/integration_plan.md` for a step-by-step implementation guide.
