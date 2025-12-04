# experiment_with_llama2_consistency.py

from utils.modelUtils import *
from utils.utils import *
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from casper import nethook
import json
import os
from tqdm import tqdm
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Initialize model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device='cuda:0'
)
print(f"Model loaded: {model_name}")
print(f"Number of layers: {mt.num_layers}")

# %%
# Test model
test_prompt = generate_input(mt.tokenizer, 'tell me a fun joke')
print("Test prompt:", test_prompt)
print("Test output:", generate_outputs(test_prompt, mt))

# %%
# Define dataset paths
jailbreaks = [
    "./datas/adversarial_amplegcg.json",
    "./datas/adversarial_autodan.json",
    "./datas/adversarial_gcg.json",
    "./datas/adversarial_pair.json"
]

fairness = [
    "./datas/adversarial_severe_toxicity.json",
    "./datas/adversarial_sexually_explicit.json",
    "./datas/adversarial_toxicity.json"
]

truthfulqa = ["./datas/truthfulqa.json"]

# Create save directories - shared with rep experiments
save_dir = "./datasets/hidden_states"
os.makedirs(save_dir, exist_ok=True)

results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# %%
def extract_all_layers_hidden_states(prompt):
    """
    Extract hidden states from all layers for all token positions.
    
    Args:
        prompt: Input prompt string
    
    Returns:
        hidden_states: Dictionary mapping layer names to hidden states [seq_len, hidden_dim]
    """
    inp = make_inputs(mt.tokenizer, [prompt])
    
    layer_names = [layername(mt.model, i) for i in range(mt.num_layers)]
    
    hidden_states = {}
    
    with torch.no_grad(), nethook.TraceDict(
        mt.model,
        layer_names,
    ) as td:
        _ = mt.model(**inp)
        
        for layer_idx in range(mt.num_layers):
            layer_name = layer_names[layer_idx]
            hidden_state = td[layer_name].output[0]
            
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]
            
            hidden_state = hidden_state[0].cpu().numpy()
            
            hidden_states[f'layer_{layer_idx}'] = hidden_state
    
    return hidden_states

# %%
def compute_layer_consistency_features(hidden_states_dict, aggregation='last'):
    """
    Compute layer consistency features using cosine similarity between consecutive layers.
    
    Args:
        hidden_states_dict: Dictionary of hidden states for all layers
        aggregation: Aggregation method - 'last' (last token) or 'mean' (mean of all tokens)
    
    Returns:
        consistency_features: Array of cosine similarities between consecutive layers [num_layers-1,]
    """
    num_layers = len(hidden_states_dict)
    consistency_features = []
    
    for layer_idx in range(num_layers - 1):
        current_layer = hidden_states_dict[f'layer_{layer_idx}']
        next_layer = hidden_states_dict[f'layer_{layer_idx + 1}']
        
        if aggregation == 'last':
            current_repr = current_layer[-1:, :]
            next_repr = next_layer[-1:, :]
        elif aggregation == 'mean':
            current_repr = current_layer.mean(axis=0, keepdims=True)
            next_repr = next_layer.mean(axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        similarity = cosine_similarity(current_repr, next_repr)[0, 0]
        consistency_features.append(similarity)
    
    return np.array(consistency_features)

# %%
def process_dataset_and_save(data_path, save_name, aggregation='last'):
    """
    Process dataset and save consistency features.
    
    Args:
        data_path: Path to input data file
        save_name: Name for saved feature file
        aggregation: Aggregation method ('last' or 'mean')
    """
    save_path = os.path.join(save_dir, f"{save_name}_{aggregation}_consistency.pkl")
    
    if os.path.exists(save_path):
        print(f"Consistency features already exist for {save_name} ({aggregation}), skipping...")
        return
    
    print(f"Processing {save_name} with aggregation={aggregation}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    results = {
        'benign': [],
        'adversarial': [],
        'aggregation': aggregation
    }
    
    benign_samples = data.get('benign', [])
    for item in tqdm(benign_samples, desc=f"{save_name} ({aggregation}) - Benign"):
        if isinstance(item, dict):
            prompt = item.get('prompt', item.get('text', ''))
        else:
            prompt = str(item)
        
        if not prompt:
            continue
        
        try:
            formatted_prompt = generate_input(mt.tokenizer, prompt)
            hidden_states = extract_all_layers_hidden_states(formatted_prompt)
            consistency_features = compute_layer_consistency_features(hidden_states, aggregation)
            
            results['benign'].append({
                'prompt': prompt,
                'consistency_features': consistency_features
            })
        except Exception as e:
            print(f"Error processing benign sample: {e}")
            continue
    
    adversarial_samples = data.get('adversarial', [])
    for item in tqdm(adversarial_samples, desc=f"{save_name} ({aggregation}) - Adversarial"):
        if isinstance(item, dict):
            prompt = item.get('prompt', item.get('text', ''))
        else:
            prompt = str(item)
        
        if not prompt:
            continue
        
        try:
            formatted_prompt = generate_input(mt.tokenizer, prompt)
            hidden_states = extract_all_layers_hidden_states(formatted_prompt)
            consistency_features = compute_layer_consistency_features(hidden_states, aggregation)
            
            results['adversarial'].append({
                'prompt': prompt,
                'consistency_features': consistency_features
            })
        except Exception as e:
            print(f"Error processing adversarial sample: {e}")
            continue
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved consistency features to {save_path}")
    print(f"Benign samples: {len(results['benign'])}, Adversarial samples: {len(results['adversarial'])}")
    print(f"Feature dimension: {len(results['benign'][0]['consistency_features']) if results['benign'] else 'N/A'}\n")

# %%
print("="*70)
print("STEP 1: Extracting Layer Consistency Features for all datasets")
print("="*70 + "\n")

aggregation_methods = ['last', 'mean']

for aggregation in aggregation_methods:
    print(f"\n{'='*70}")
    print(f"Aggregation Method: {aggregation}")
    print(f"{'='*70}\n")
    
    for jb_path in jailbreaks:
        dataset_name = os.path.basename(jb_path).replace('.json', '')
        process_dataset_and_save(jb_path, dataset_name, aggregation)
    
    for fair_path in fairness:
        dataset_name = os.path.basename(fair_path).replace('.json', '')
        process_dataset_and_save(fair_path, dataset_name, aggregation)
    
    for tqa_path in truthfulqa:
        dataset_name = os.path.basename(tqa_path).replace('.json', '')
        process_dataset_and_save(tqa_path, dataset_name, aggregation)

print("Layer consistency features extraction completed!\n")

# %%
def load_consistency_features(dataset_name, aggregation='last'):
    """
    Load saved consistency features from pickle file.
    
    Args:
        dataset_name: Name of the dataset
        aggregation: Aggregation method used
    
    Returns:
        data: Dictionary containing consistency features
    """
    save_path = os.path.join(save_dir, f"{dataset_name}_{aggregation}_consistency.pkl")
    
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

# %%
class LayerConsistencyClassifier:
    """
    Classifier based on layer consistency features.
    Uses cosine similarity between consecutive layers as features for classification.
    """
    def __init__(self, aggregation='last'):
        """
        Args:
            aggregation: Aggregation method ('last' or 'mean')
        """
        self.aggregation = aggregation
        self.mlp_model = None
        self.training_info = {
            'aggregation': aggregation,
            'training_dataset': None,
            'n_benign_samples': 0,
            'n_adversarial_samples': 0,
            'feature_dim': 0
        }
    
    def train(self, consistency_data, dataset_name):
        """
        Train MLP classifier on consistency features.
        
        Args:
            consistency_data: Dictionary containing benign and adversarial consistency features
            dataset_name: Name of training dataset
        """
        self.training_info['training_dataset'] = dataset_name
        self.training_info['n_benign_samples'] = len(consistency_data['benign'])
        self.training_info['n_adversarial_samples'] = len(consistency_data['adversarial'])
        
        X_train = []
        y_train = []
        
        for item in consistency_data['benign']:
            X_train.append(item['consistency_features'])
            y_train.append(0)
        
        for item in consistency_data['adversarial']:
            X_train.append(item['consistency_features'])
            y_train.append(1)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.training_info['feature_dim'] = X_train.shape[1]
        
        print(f"Training MLP model on consistency features...")
        print(f"  Feature dimension: {X_train.shape[1]} (num_layers - 1)")
        print(f"  Training samples: {len(X_train)}")
        
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
        self.mlp_model.fit(X_train, y_train)
        
        train_pred = self.mlp_model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='binary')
        
        self.training_info['train_accuracy'] = float(train_acc)
        self.training_info['train_f1'] = float(train_f1)
        
        print(f"  Training accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    
    def predict(self, consistency_data):
        """
        Make predictions on consistency features.
        
        Args:
            consistency_data: Dictionary containing consistency features
        
        Returns:
            predictions: Predicted labels
            true_labels: True labels
        """
        X_test = []
        y_test = []
        
        for item in consistency_data['benign']:
            X_test.append(item['consistency_features'])
            y_test.append(0)
        
        for item in consistency_data['adversarial']:
            X_test.append(item['consistency_features'])
            y_test.append(1)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        predictions = self.mlp_model.predict(X_test)
        
        return predictions, y_test

# %%
def run_consistency_experiments(dataset_names, aggregation='last'):
    """
    Run layer consistency classification experiments across datasets.
    
    Args:
        dataset_names: List of dataset names
        aggregation: Aggregation method
    
    Returns:
        all_results: Dictionary containing all experiment results
    """
    all_results = {}
    
    for train_idx, train_name in enumerate(dataset_names):
        print(f"\n{'='*70}")
        print(f"Training on: {train_name} (aggregation={aggregation})")
        print(f"{'='*70}")
        
        train_data = load_consistency_features(train_name, aggregation)
        
        classifier = LayerConsistencyClassifier(aggregation)
        classifier.train(train_data, train_name)
        
        test_results = {}
        
        for test_name in dataset_names:
            try:
                test_data = load_consistency_features(test_name, aggregation)
                y_pred, y_test = classifier.predict(test_data)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='binary')
                
                test_results[test_name] = {
                    'accuracy': accuracy,
                    'f1': f1
                }
                
                marker = " (SAME)" if train_name == test_name else ""
                print(f"  {test_name:40s}{marker:8s} - Acc: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"  Error with {test_name}: {e}")
                test_results[test_name] = {
                    'accuracy': 0.0,
                    'f1': 0.0
                }
        
        all_results[train_name] = {
            'training_info': classifier.training_info.copy(),
            'test_results': test_results
        }
    
    return all_results

# %%
jailbreak_names = [os.path.basename(p).replace('.json', '') for p in jailbreaks]
fairness_names = [os.path.basename(p).replace('.json', '') for p in fairness]
truthfulqa_name = os.path.basename(truthfulqa[0]).replace('.json', '')

# %%
all_experiment_results = {}

# %%
print("\n" + "="*70)
print("STEP 2: Running Jailbreak Experiments")
print("="*70)

for aggregation in aggregation_methods:
    print(f"\n{'#'*70}")
    print(f"# Configuration: aggregation={aggregation}")
    print(f"{'#'*70}")
    
    config_key = f"jailbreak_{aggregation}"
    all_experiment_results[config_key] = run_consistency_experiments(
        jailbreak_names, aggregation
    )

# %%
print("\n" + "="*70)
print("STEP 3: Running Fairness Experiments")
print("="*70)

for aggregation in aggregation_methods:
    print(f"\n{'#'*70}")
    print(f"# Configuration: aggregation={aggregation}")
    print(f"{'#'*70}")
    
    config_key = f"fairness_{aggregation}"
    all_experiment_results[config_key] = run_consistency_experiments(
        fairness_names, aggregation
    )

# %%
print("\n" + "="*70)
print("STEP 4: Running TruthfulQA Experiments")
print("="*70)

for aggregation in aggregation_methods:
    print(f"\n{'#'*70}")
    print(f"# Configuration: aggregation={aggregation}")
    print(f"{'#'*70}")
    
    truthfulqa_data = load_consistency_features(truthfulqa_name, aggregation)
    
    benign_count = len(truthfulqa_data['benign'])
    adversarial_count = len(truthfulqa_data['adversarial'])
    
    split_idx_benign = benign_count // 2
    split_idx_adversarial = adversarial_count // 2
    
    train_data = {
        'benign': truthfulqa_data['benign'][:split_idx_benign],
        'adversarial': truthfulqa_data['adversarial'][:split_idx_adversarial],
        'aggregation': aggregation
    }
    
    test_data = {
        'benign': truthfulqa_data['benign'][split_idx_benign:],
        'adversarial': truthfulqa_data['adversarial'][split_idx_adversarial:],
        'aggregation': aggregation
    }
    
    print(f"Training samples: {len(train_data['benign']) + len(train_data['adversarial'])}")
    print(f"Test samples: {len(test_data['benign']) + len(test_data['adversarial'])}")
    
    classifier = LayerConsistencyClassifier(aggregation)
    classifier.train(train_data, truthfulqa_name)
    
    y_pred, y_test = classifier.predict(test_data)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    config_key = f"truthfulqa_{aggregation}"
    all_experiment_results[config_key] = {
        'training_info': classifier.training_info.copy(),
        'accuracy': accuracy,
        'f1': f1
    }
    
    print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# %%
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70 + "\n")

print("JAILBREAK RESULTS:")
print("-" * 70)
for aggregation in aggregation_methods:
    config_key = f"jailbreak_{aggregation}"
    if config_key in all_experiment_results:
        print(f"\nAggregation: {aggregation}")
        for train_name, results in all_experiment_results[config_key].items():
            print(f"  Trained on: {train_name}")
            for test_name, metrics in results['test_results'].items():
                marker = " (SAME)" if train_name == test_name else ""
                print(f"    {test_name:40s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*70)
print("FAIRNESS RESULTS:")
print("-" * 70)
for aggregation in aggregation_methods:
    config_key = f"fairness_{aggregation}"
    if config_key in all_experiment_results:
        print(f"\nAggregation: {aggregation}")
        for train_name, results in all_experiment_results[config_key].items():
            print(f"  Trained on: {train_name}")
            for test_name, metrics in results['test_results'].items():
                marker = " (SAME)" if train_name == test_name else ""
                print(f"    {test_name:40s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*70)
print("TRUTHFULQA RESULTS:")
print("-" * 70)
for aggregation in aggregation_methods:
    config_key = f"truthfulqa_{aggregation}"
    if config_key in all_experiment_results:
        result = all_experiment_results[config_key]
        print(f"\nAggregation: {aggregation}")
        print(f"  Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")

print("\n" + "="*70)
print("Experiment completed!")
print("="*70)

# %%
results_file = os.path.join(results_dir, 'llama2_layer_consistency_results.json')

final_results = {
    'jailbreak': {},
    'fairness': {},
    'truthfulqa': {}
}

for aggregation in aggregation_methods:
    jb_key = f"jailbreak_{aggregation}"
    if jb_key in all_experiment_results:
        final_results['jailbreak'][aggregation] = {}
        for train_name, results in all_experiment_results[jb_key].items():
            final_results['jailbreak'][aggregation][train_name] = {
                'training_info': results['training_info'],
                'test_results': {k: {'accuracy': float(v['accuracy']), 'f1': float(v['f1'])} 
                               for k, v in results['test_results'].items()}
            }
    
    fair_key = f"fairness_{aggregation}"
    if fair_key in all_experiment_results:
        final_results['fairness'][aggregation] = {}
        for train_name, results in all_experiment_results[fair_key].items():
            final_results['fairness'][aggregation][train_name] = {
                'training_info': results['training_info'],
                'test_results': {k: {'accuracy': float(v['accuracy']), 'f1': float(v['f1'])} 
                               for k, v in results['test_results'].items()}
            }
    
    tqa_key = f"truthfulqa_{aggregation}"
    if tqa_key in all_experiment_results:
        final_results['truthfulqa'][aggregation] = {
            'training_info': all_experiment_results[tqa_key]['training_info'],
            'accuracy': float(all_experiment_results[tqa_key]['accuracy']),
            'f1': float(all_experiment_results[tqa_key]['f1'])
        }

with open(results_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\nAll results saved to: {results_file}")