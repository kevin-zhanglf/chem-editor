# Explainability and Auditability for Polymer Design Agents

## Overview

This document describes techniques and best practices for making AI-driven polymer design workflows transparent, interpretable, and auditable. When integrating AgentEvolver into materials discovery, explainability is critical for:

- **Trust**: Materials scientists must understand why the agent recommends certain polymers
- **Debugging**: Identify when agents or surrogate models are making errors
- **Learning**: Extract generalizable design principles from agent behavior
- **Compliance**: Satisfy regulatory and safety requirements with audit trails
- **Collaboration**: Enable human-agent teaming with clear communication

## Explainability Techniques

### 1. SHAP (SHapley Additive exPlanations)

**Purpose**: Quantify the contribution of each structural feature to predicted properties.

**How It Works**:
- SHAP values assign credit to input features (e.g., monomer types, block lengths) for a prediction
- Based on cooperative game theory (Shapley values)
- Model-agnostic: works with any surrogate model (neural nets, random forests, etc.)

**Implementation**:
```python
import shap

# Assume surrogate_model predicts Tg from BigSMILES encoding
explainer = shap.Explainer(surrogate_model, background_data)
shap_values = explainer(polymer_encoding)

# Visualize feature importance
shap.plots.waterfall(shap_values[0])  # For a single polymer
shap.plots.beeswarm(shap_values)      # For multiple polymers
```

**Outputs**:
- Feature importance rankings: "Aromatic monomers contribute +15°C to Tg"
- Interaction effects: "Combining monomer A and B increases dielectric constant"
- Visualizations: Waterfall plots, force plots, summary plots

**Use Cases**:
- Explain why agent-proposed polymer X has high predicted Tg
- Identify which structural changes most impact cost or feasibility
- Validate surrogate model behavior on edge cases

**References**:
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Original Paper](https://arxiv.org/abs/1705.07874)

---

### 2. Local Surrogate Models (LIME)

**Purpose**: Approximate a complex model's behavior locally with an interpretable model.

**How It Works**:
- Perturb the input polymer structure (change monomers, block lengths)
- Query the surrogate model on perturbed samples
- Fit a simple linear model to explain the local behavior

**Implementation**:
```python
from lime.lime_tabular import LimeTabularExplainer

# Assume polymer is encoded as a feature vector
explainer = LimeTabularExplainer(
    training_data=training_encodings,
    mode="regression",
    feature_names=feature_names
)

# Explain a single prediction
explanation = explainer.explain_instance(
    polymer_encoding,
    surrogate_model.predict,
    num_features=10
)

explanation.show_in_notebook()  # Visualize top features
```

**Outputs**:
- Top-K features influencing the prediction
- Linear coefficients: "If feature X increases by 1, Tg increases by 5°C"

**Use Cases**:
- Quickly understand local behavior without retraining surrogate models
- Validate surrogate predictions before experimental synthesis
- Identify features that the model relies on (may reveal biases)

**References**:
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Original Paper](https://arxiv.org/abs/1602.04938)

---

### 3. Counterfactual Explanations

**Purpose**: Answer "what-if" questions by identifying minimal changes that flip a decision.

**How It Works**:
- Find the smallest modification to a polymer that achieves the target property
- Constrain changes to be chemically valid (e.g., swap monomers, adjust ratios)

**Example**:
- Current: "Polymer A has Tg = 120°C (below target of 150°C)"
- Counterfactual: "Replacing monomer X with aromatic monomer Y increases Tg to 155°C"

**Implementation**:
```python
def generate_counterfactual(current_polymer, target_property, surrogate_model):
    """
    Search for minimal mutation that achieves target property.
    """
    best_polymer = None
    min_distance = float('inf')
    
    for mutation in mutation_operators:
        candidate = mutation(current_polymer)
        predicted = surrogate_model.predict(candidate)
        
        if predicted >= target_property:
            distance = compute_structural_distance(current_polymer, candidate)
            if distance < min_distance:
                best_polymer = candidate
                min_distance = distance
    
    return best_polymer, min_distance
```

**Outputs**:
- Minimal-change polymer that meets targets
- Structural differences highlighted (diff visualization)

**Use Cases**:
- Suggest actionable modifications to materials scientists
- Validate agent's mutation strategies
- Identify decision boundaries in property space

**References**:
- [Counterfactual Explanations Survey](https://arxiv.org/abs/2010.10596)
- [DiCE Library](https://github.com/interpretml/DiCE)

---

### 4. Rule Extraction from Agent Trajectories

**Purpose**: Convert learned policies into human-readable design rules.

**How It Works**:
- Analyze agent trajectories: (state, action, reward) sequences
- Cluster high-reward trajectories to identify common patterns
- Extract rules: "When Tg is below target, add aromatic monomers 80% of the time"

**Implementation**:
```python
def extract_rules(trajectories, min_support=0.6):
    """
    Extract decision rules from agent trajectories using association mining.
    """
    rules = []
    
    # Group trajectories by state conditions
    for condition in state_conditions:
        matching_trajectories = filter_by_condition(trajectories, condition)
        action_counts = count_actions(matching_trajectories)
        
        # Create rule if action frequency exceeds threshold
        for action, count in action_counts.items():
            frequency = count / len(matching_trajectories)
            if frequency >= min_support:
                rule = {
                    'condition': condition,
                    'action': action,
                    'frequency': frequency,
                    'avg_reward': compute_avg_reward(matching_trajectories, action)
                }
                rules.append(rule)
    
    return sorted(rules, key=lambda r: r['avg_reward'], reverse=True)
```

**Outputs**:
- Decision tree or rule list: "IF Tg < 150 AND cost > 100 THEN substitute expensive monomer"
- Confidence scores and applicability ranges

**Use Cases**:
- Codify agent knowledge into design guidelines for materials scientists
- Identify when agent is using heuristics vs. exploration
- Validate agent behavior against domain expertise

**References**:
- [Apprenticeship Learning](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)
- [Policy Distillation](https://arxiv.org/abs/1511.06295)

---

### 5. Attention Visualization (for Neural Surrogate Models)

**Purpose**: Visualize which parts of the polymer structure the model focuses on.

**How It Works**:
- If using Transformer-based models (e.g., SMILES encoders), extract attention weights
- Overlay attention scores on the BigSMILES string or molecular graph

**Example**:
```
BigSMILES: {[>][<]CC(C)(C)[>]}[<]

Attention weights:
  CC(C)(C)  ▰▰▰▰▰▰▰▰▰▰  (high attention - branched structure affects Tg)
  [>][<]    ▰▰▰         (low attention - connection points less relevant)
```

**Implementation**:
- Use Captum, Grad-CAM, or model-specific attention extraction
- Visualize with heatmaps or overlays on molecular structures

**Use Cases**:
- Verify model is focusing on chemically meaningful features
- Debug model biases (e.g., overfitting to dataset artifacts)
- Guide feature engineering for non-neural models

**References**:
- [Captum Library](https://captum.ai/)
- [Attention in Chemistry](https://doi.org/10.1021/acs.jcim.0c00479)

---

### 6. Uncertainty Quantification

**Purpose**: Quantify confidence in surrogate model predictions.

**How It Works**:
- Use Bayesian neural networks, Monte Carlo dropout, or ensemble methods
- Report prediction intervals: "Tg = 150 ± 10°C (95% confidence)"

**Implementation**:
```python
# Monte Carlo Dropout example
def predict_with_uncertainty(polymer, model, n_samples=100):
    model.train()  # Enable dropout at inference time
    predictions = []
    
    for _ in range(n_samples):
        pred = model(polymer)
        predictions.append(pred)
    
    mean = np.mean(predictions)
    std = np.std(predictions)
    
    return mean, std

# Usage
tg_mean, tg_std = predict_with_uncertainty(polymer_encoding, tg_model)
print(f"Tg = {tg_mean:.1f} ± {tg_std:.1f}°C")
```

**Outputs**:
- Prediction mean and standard deviation
- Confidence intervals for decision-making

**Use Cases**:
- Prioritize experimental validation (high uncertainty = high information gain)
- Flag out-of-distribution polymers that may be risky
- Improve active learning acquisition functions

**References**:
- [Uncertainty in Deep Learning](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html)
- [Ensemble Methods Survey](https://arxiv.org/abs/1106.0257)

---

## Auditability and Landing Recommendations

### Audit Trail Requirements

To ensure reproducibility and compliance, log the following for every agent decision:

1. **Agent State**:
   - Current polymer BigSMILES string
   - Predicted property values from surrogate models
   - Uncertainty estimates

2. **Agent Action**:
   - Mutation operator applied
   - Parameters (e.g., which monomer substituted, new block length)
   - Timestamp and episode number

3. **Environment Response**:
   - New polymer BigSMILES string after mutation
   - Updated property predictions
   - Reward value
   - Validity and feasibility flags

4. **Model Metadata**:
   - Surrogate model version and checkpoint
   - RL policy version and hyperparameters
   - LLM planner model and prompt template (if used)

5. **Experimental Validation** (when applicable):
   - Synthesis protocol
   - Measured property values
   - Comparison with surrogate predictions (error analysis)
   - Lab technician notes and conditions

### Recommended Audit Log Format

Use structured logging (JSON or database) for downstream analysis:

```json
{
  "timestamp": "2025-11-24T06:25:00Z",
  "episode_id": "ep_12345",
  "step": 42,
  "agent": {
    "policy_version": "v2.3",
    "model_checkpoint": "rl_agent_step500k.pt"
  },
  "state": {
    "bigsmiles": "{[>][<]CC(C)(C)[>]}[<]",
    "predicted_tg": 145.2,
    "predicted_dielectric": 3.8,
    "predicted_cost": 12.5,
    "uncertainty_tg": 8.1
  },
  "action": {
    "operator": "mutate_substitute",
    "params": {"position": 2, "new_monomer": "aromatic_A"}
  },
  "next_state": {
    "bigsmiles": "{[>][<]c1ccccc1[>]}[<]",
    "predicted_tg": 162.7,
    "predicted_dielectric": 4.1,
    "predicted_cost": 15.0
  },
  "reward": 0.85,
  "done": false,
  "info": {
    "validity": true,
    "feasibility": true,
    "constraint_violations": []
  }
}
```

### Storage and Retrieval

- Store logs in a database (PostgreSQL, MongoDB) or data lake (S3, BigQuery)
- Index by episode_id, timestamp, polymer BigSMILES for fast queries
- Retain logs for at least 1-2 years for compliance and retraining

### Explainability Reports for Experimental Team

Generate human-readable reports for high-priority candidates:

**Example Report**:

```
=== Polymer Candidate #47 ===

BigSMILES: {[>][<]c1ccccc1C(C)(C)[>]}[<]

Predicted Properties:
  - Glass Transition Temperature (Tg): 162.7 ± 8.1°C ✓ (Target: >150°C)
  - Dielectric Constant: 4.1 ± 0.3 ✓ (Target: <5)
  - Estimated Cost: $15.00/kg ✓ (Target: <$20/kg)

Why this polymer was selected:
  - Agent applied 'mutate_substitute' to add aromatic rings
  - SHAP analysis: Aromatic groups contribute +18°C to Tg
  - High reward (0.85) due to meeting all target properties
  - Moderate uncertainty (8.1°C) → good candidate for validation

Suggested synthesis route:
  - Use commercially available monomer X (aromatic_A)
  - Polymerization conditions: [user to fill in]

Predicted risks:
  - None detected (passes validity and feasibility checks)

Next steps:
  1. Synthesize 10g sample
  2. Measure Tg using DSC
  3. Update surrogate model with experimental result
```

### Explainability Dashboard (Optional)

Build a web dashboard for interactive exploration:

- **Trajectory Viewer**: Visualize agent's exploration path in property space
- **Feature Importance**: Dynamic SHAP plots for any polymer
- **Counterfactual Generator**: What-if tool for materials scientists
- **Rule Browser**: Search extracted design rules by condition and action
- **Audit Log Search**: Query and filter historical agent decisions

**Recommended Tools**: Streamlit, Dash, or Gradio for rapid prototyping

---

## Best Practices

1. **Explainability First**: Design explainability into the workflow from day one, not as an afterthought
2. **Domain Validation**: Always validate explanations with materials scientists (do they make chemical sense?)
3. **Diverse Techniques**: Use multiple explainability methods (SHAP + counterfactuals + rules) for robustness
4. **Uncertainty Awareness**: Treat predictions with high uncertainty as exploratory hypotheses
5. **Human-in-the-Loop**: Allow experts to override agent decisions and provide feedback
6. **Continuous Monitoring**: Track explanation quality over time (e.g., do SHAP values align with experimental results?)
7. **Document Failures**: Log cases where agent or surrogate models fail, and analyze root causes
8. **Simplicity**: Prefer simple, actionable explanations over complex technical details for non-ML users

---

## Integration with Active Learning

Explainability techniques can improve active learning efficiency:

- **SHAP-based Acquisition**: Select candidates with high SHAP value diversity (explore different mechanisms)
- **Counterfactual Sampling**: Validate decision boundaries by testing edge cases
- **Rule-based Filtering**: Use extracted rules to pre-screen candidates before expensive computations
- **Uncertainty-guided Exploration**: Prioritize regions where surrogate models are least confident

---

## Security and Privacy Considerations

- **Sensitive Data**: Redact proprietary polymer structures or experimental data in public logs
- **Model Stealing**: Limit access to surrogate model APIs to prevent reverse engineering
- **Audit Access Control**: Restrict audit log access to authorized personnel only
- **Anonymization**: Remove personally identifiable information (PII) from logs if applicable

---

## References and Further Reading

- [Explainable AI for Materials Science](https://doi.org/10.1038/s43588-021-00150-3)
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)
- [DARPA XAI Program](https://www.darpa.mil/program/explainable-artificial-intelligence)
- [EU AI Act (Transparency Requirements)](https://artificialintelligenceact.eu/)

---

## Next Steps

1. Choose 2-3 explainability techniques to implement in Phase 2-3 (recommend SHAP + counterfactuals)
2. Design audit log schema and set up logging infrastructure
3. Validate explanations with materials scientists in Phase 4 (closed-loop validation)
4. Build explainability reports for experimental team in Phase 5 (productionization)
5. Consider developing a dashboard for interactive exploration (optional, Phase 5)

For integration details, see `docs/integration_plan.md`.  
For system architecture, see `docs/architecture.md`.
