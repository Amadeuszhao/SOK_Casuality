# LLM Causality Analysis Framework

A comprehensive framework for multi-level causality analysis in Large Language Models (LLMs), enabling systematic investigation of safety mechanisms and misbehavior detection across token, neuron, layer, and representation levels.

## 📋 Overview

This framework implements the causality analysis methods described in "[SoK: a Comprehensive Causality Analysis Framework for Large Language Model Security](https://arxiv.org/abs/2512.04841)". It provides tools for:

- **Multi-level Causality Analysis**: Systematic interventions at token, neuron, layer, and representation levels
- **Misbehavior Detection**: Detection of jailbreaks, hallucinations, backdoors, and fairness violations using causal features
- **Flexible Architecture**: Extensible design supporting multiple LLM architectures and safety-critical tasks

## 🏗️ Project Structure

```
.
├── datas/              # Raw datasets for evaluation
├── datasets/           # Generated causal features
│   ├── hidden_states/  # Representation-level features
│   ├── neuron/         # Neuron-level features
│   ├── layers/         # Layer-level features
│   └── token/          # Token-level features
├── casper/             # Core causality analysis engine
├── utils/              # Utility functions and helpers
├── results/            # Detection results and evaluation metrics
├── experiment_with_layer.ipynb    # Layer-level causality experiments
├── experiment_with_neuron.ipynb   # Neuron-level causality experiments
├── experiment_with_rep.ipynb      # Representation-level experiments
└── experiment_with_token.ipynb    # Token-level causality experiments
```

## 🔬 Causality Analysis & Experimental Findings

### 1. Token-Level Analysis
Examines how individual input tokens causally affect model outputs through systematic replacement interventions.

**Intervention**: `do(xi = [PAD])`

**Key Findings**:
- GCG attacks: High token-level ACE (~0.7), concentrated in specific triggers
- AutoDAN attacks: Lower ACE (max 0.15), semantically distributed
- **Intervention Efficacy**: ASR reduced from 100% → 24.2-30.4% for GCG by removing single tokens

**Experimentation** (`experiment_with_token.ipynb`):
- Test different types of adversarial prompts (GCG, AutoDAN, PAIR, AmpleGCG)
- Analyze token-level causal patterns across various jailbreak strategies
- Compare trigger token effectiveness between optimization-based and LLM-based attacks

### 2. Neuron-Level Analysis
Identifies sparse, causally critical neurons that modulate safety behavior using logistic regression and z-score analysis.

**Intervention**: `do(neuron_i = 0)`

**Key Findings**:
- Only 0.88-1.88% of neurons per layer are safety-critical
- High cross-model transferability
- **Intervention Efficacy**: ASR increased from 0% → 46.8-57.8% on harmful prompts by deactivating safety neurons

**Experimentation** (`experiment_with_neuron.ipynb`):
- **Customize training datasets**: Use different benign and harmful datasets to train neuron classifiers
- **Locate toxic neurons**: Identify safety-critical neurons through z-score analysis (|z| > 2.5)
- **Cross-dataset validation**: Test neuron transferability across different prompt distributions
- **Layer-wise analysis**: Examine neuron sparsity patterns across transformer layers

### 3. Layer-Level Analysis
Traces how causal influence propagates through transformer layers to shape safety decisions.

**Intervention**: `do(remove layers [ℓ, ℓ+n])`

**Key Findings**:
- Safety mechanisms concentrate in early-to-middle layers (2-12)
- Layer 2 shows maximum causal effect (~76% ACE)
- Negligible effects after Layer 13
- **Intervention Efficacy**: ASR increased from 0% → 91.4-92.8% by removing critical layers

**Experimentation** (`experiment_with_layer.ipynb`):
- **Switch harmful prompts**: Experiment with different harmful prompt sets to identify safety-critical layers
- **Discover layer patterns**: Analyze how layer importance varies across different threat models
- **Validate layer localization**: Confirm whether safety discriminators consistently appear in early layers
- **Cross-prompt consistency**: Test layer causality stability across diverse harmful content

### 4. Representation-Level Analysis
Explores how embedding geometry encodes safety boundaries through directional interventions.

**Intervention**: `do(h^(ℓ) = h^(ℓ) + 0.5·||h^(ℓ)||₂·e_a)`

**Key Findings**:
- Clear geometric separation between harmful/benign clusters in representation space
- Safety alignment operates through geometrically separable structures
- **Intervention Efficacy**: ASR increased from 0% → 92.8-96.0% (highest efficacy across all levels)

**Experimentation** (`experiment_with_rep.ipynb`):
- **Customize datasets**: Use different benign and harmful datasets to compute direction vectors
- **Control intervention layers**: Specify which layers to apply directional steering
- **Adjust steering strength**: Modify the magnitude of directional perturbations (alpha parameter)
- **Visualize decision boundaries**: Use PCA to analyze representation space geometry
- **Multi-layer steering**: Experiment with layer-specific or uniform steering across the model

## 🎯 Detection Tasks & Performance

The framework supports detection across four categories of model misbehavior:

### 1. Jailbreak Detection
- **Attacks**: GCG, AutoDAN, AmpleGCG, PAIR
- **Dataset**: AdvBench (500 prompts per attack)
- **Best Performance**: 
  - Neuron-level: F1: 0.977-0.994, DSR: 100%
  - Representation-level: F1: 0.946-0.992

### 2. Hallucination Detection
- **Dataset**: TruthfulQA
- **Best Performance**: Multi-level fusion (F1: 0.956-0.987, DSR: 97-100%)
- **Note**: Single-level methods achieve <0.7 F1; feature fusion essential

### 3. Backdoor Detection
- **Attacks**: BadNets, CTBA, MTBA, Sleeper Agent
- **Dataset**: 500 trigger-embedded prompts per method
- **Best Performance**: Representation-level (F1: 0.923-0.992, DSR: 94-99.6%)

### 4. Fairness Violation Detection
- **Dataset**: RealToxicityPrompts (toxicity, sexually explicit, severe toxicity)
- **Best Performance**: Neuron-level (F1: 0.990-1.000, DSR: 100%)

### Performance Summary

| Analysis Level | Detection Time | Jailbreak F1 | Backdoor F1 | Fairness F1 |
|----------------|----------------|--------------|-------------|-------------|
| **Neuron** | 0.12-0.14s | **0.977-0.994** | 0.939-0.983 | **0.990-1.000** |
| **Representation** | 0.07-0.10s | 0.946-0.992 | **0.952-0.992** | 0.967-0.988 |
| Layer | 1.92-2.85s | 0.712-0.923 | 0.646-0.979 | 0.802-0.896 |
| Token | 2.87-4.37s | 0.430-0.881 | 0.564-0.909 | 0.607-0.714 |

**Key Insights**:
- **Neuron-level** and **Representation-level** methods offer the best balance of accuracy and efficiency
- Multi-level feature fusion significantly improves hallucination detection
- Single-pass methods (Neuron, Representation) are 20-60× faster than intervention-based methods

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Supported Models
- LLaMA2-7B
- Qwen2.5-7B
- LLaMA3.1-8B

### Running Experiments

Each Jupyter notebook provides interactive experimentation for different causality analysis levels:

#### Token-Level Experiments
```bash
jupyter notebook experiment_with_token.ipynb
```
- Experiment with different adversarial prompt types
- Analyze causal token patterns across attack methods

#### Layer-Level Experiments
```bash
jupyter notebook experiment_with_layer.ipynb
```
- Switch between different harmful prompt sets
- Identify safety-critical layers for various threat models

#### Neuron-Level Experiments
```bash
jupyter notebook experiment_with_neuron.ipynb
```
- Customize benign/harmful training datasets
- Locate and analyze toxic neurons across layers

#### Representation-Level Experiments
```bash
jupyter notebook experiment_with_rep.ipynb
```
- Configure benign/harmful datasets for direction computation
- Control intervention layers and steering strength
- Visualize representation space geometry

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Support for additional LLM architectures
- New causality-based detection methods
- Efficiency optimizations for large-scale deployment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Responsible Use

This framework is designed for research purposes to improve LLM safety. Users should:
- Obtain proper authorization before testing on production systems
- Follow responsible disclosure practices for discovered vulnerabilities
- Use detection capabilities to enhance model safety, not to facilitate attacks

---

**Status**: Under Active Development | **Last Updated**: December 2025