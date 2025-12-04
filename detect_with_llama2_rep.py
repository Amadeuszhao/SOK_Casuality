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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from itertools import combinations

# %%
# Initialize model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=None,
    device='cuda:0'
)
mt.model

# %%
# Test prompt generation
test_prompt = generate_input(mt.tokenizer, 'tell me a fun joke')
test_prompt

# %%
print("Test Generate Output", generate_outputs(test_prompt, mt))

# %%
def extract_hidden_states(prompt, layer_idx):
    """
    Extract hidden states from specified layer.
    
    Args:
        prompt: Input prompt string
        layer_idx: Layer index to extract from
    
    Returns:
        hidden_state: Hidden state from last token position (numpy array)
    """
    inp = make_inputs(mt.tokenizer, [prompt])
    layer_name = layername(mt.model, layer_idx)
    
    with torch.no_grad(), nethook.TraceDict(
        mt.model,
        [layer_name],
    ) as td:
        _ = mt.model(**inp)
        hidden_state = td[layer_name].output[0]
        
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]
        
        hidden_state = hidden_state[0, -1, :].cpu().numpy()
    
    return hidden_state

# %%
def extract_all_layers_hidden_states(prompt):
    """
    Extract hidden states from all layers for a given prompt.
    
    Args:
        prompt: Input prompt string
    
    Returns:
        hidden_states: Dictionary mapping layer names to their hidden states
    """
    hidden_states = {}
    
    for layer_idx in range(mt.num_layers):
        hidden_states[f'layer_{layer_idx}'] = extract_hidden_states(prompt, layer_idx)
    
    return hidden_states

# %%
# Test the function
test_hidden_states = extract_all_layers_hidden_states(test_prompt)
print(f"Extracted hidden states from {len(test_hidden_states)} layers")
print(f"Hidden state shape for layer 0: {test_hidden_states['layer_0'].shape}")

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

# Create save directory
save_dir = "./datasets/hidden_states"
os.makedirs(save_dir, exist_ok=True)

# %%
def extract_and_save_hidden_states(data_path, save_name):
    """
    Extract hidden states from all layers for dataset and save to pickle file.
    
    Args:
        data_path: Path to input data file
        save_name: Name for saved feature file
    """
    save_path = os.path.join(save_dir, f"{save_name}.pkl")
    
    if os.path.exists(save_path):
        print(f"Hidden states already exist for {save_name}, skipping...")
        return
    
    print(f"Processing {save_name}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    results = {
        'benign': [],
        'adversarial': []
    }
    
    benign_samples = data.get('benign', [])
    for item in tqdm(benign_samples, desc=f"{save_name} - Benign"):
        if isinstance(item, dict):
            prompt = item.get('prompt', item.get('text', ''))
        else:
            prompt = str(item)
        
        if not prompt:
            continue
        
        try:
            formatted_prompt = generate_input(mt.tokenizer, prompt)
            hidden_states = extract_all_layers_hidden_states(formatted_prompt)
            
            results['benign'].append({
                'prompt': prompt,
                'hidden_states': hidden_states
            })
        except Exception as e:
            print(f"Error processing benign sample: {e}")
            continue
    
    adversarial_samples = data.get('adversarial', [])
    for item in tqdm(adversarial_samples, desc=f"{save_name} - Adversarial"):
        if isinstance(item, dict):
            prompt = item.get('prompt', item.get('text', ''))
        else:
            prompt = str(item)
        
        if not prompt:
            continue
        
        try:
            formatted_prompt = generate_input(mt.tokenizer, prompt)
            hidden_states = extract_all_layers_hidden_states(formatted_prompt)
            
            results['adversarial'].append({
                'prompt': prompt,
                'hidden_states': hidden_states
            })
        except Exception as e:
            print(f"Error processing adversarial sample: {e}")
            continue
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved hidden states to {save_path}")
    print(f"Benign samples: {len(results['benign'])}, Adversarial samples: {len(results['adversarial'])}\n")

# %%
print("="*60)
print("STEP 1: Extracting and saving hidden states for all datasets")
print("="*60 + "\n")

for jb_path in jailbreaks:
    dataset_name = os.path.basename(jb_path).replace('.json', '')
    extract_and_save_hidden_states(jb_path, dataset_name)

for fair_path in fairness:
    dataset_name = os.path.basename(fair_path).replace('.json', '')
    extract_and_save_hidden_states(fair_path, dataset_name)

for tqa_path in truthfulqa:
    dataset_name = os.path.basename(tqa_path).replace('.json', '')
    extract_and_save_hidden_states(tqa_path, dataset_name)

print("Hidden states extraction completed!\n")

# %%
def load_hidden_states(pickle_path):
    """
    Load hidden states from pickle file.
    
    Returns:
        data: Dictionary containing benign and adversarial hidden states
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

# %%
class PCADistanceClassifier:
    """
    PCA-based distance classifier:
    1. Apply PCA dimensionality reduction to hidden states
    2. Calculate benign and adversarial center points in PCA space
    3. For each sample, compute distances to both centers as features
    4. Train MLP classifier on distance features
    """
    def __init__(self, n_components=50, layer_indices=None):
        """
        Args:
            n_components: Number of PCA components
            layer_indices: List of layer indices to use
        """
        self.n_components = n_components
        self.layer_indices = layer_indices
        self.pca_models = {}
        self.benign_centers = {}
        self.adversarial_centers = {}
        self.training_info = {
            'n_components': n_components,
            'layer_indices': layer_indices,
            'training_dataset': None,
            'n_benign_samples': 0,
            'n_adversarial_samples': 0
        }
    
    def fit(self, hidden_states_data, dataset_name):
        """
        Train PCA models and calculate center points for each layer.
        
        Args:
            hidden_states_data: Dictionary containing benign and adversarial hidden states
            dataset_name: Name of training dataset
        """
        self.training_info['training_dataset'] = dataset_name
        self.training_info['n_benign_samples'] = len(hidden_states_data['benign'])
        self.training_info['n_adversarial_samples'] = len(hidden_states_data['adversarial'])
        
        for layer_idx in self.layer_indices:
            layer_key = f'layer_{layer_idx}'
            
            benign_states = []
            adversarial_states = []
            
            for item in hidden_states_data['benign']:
                benign_states.append(item['hidden_states'][layer_key])
            
            for item in hidden_states_data['adversarial']:
                adversarial_states.append(item['hidden_states'][layer_key])
            
            benign_states = np.array(benign_states)
            adversarial_states = np.array(adversarial_states)
            
            all_states = np.vstack([benign_states, adversarial_states])
            
            pca = PCA(n_components=self.n_components)
            pca.fit(all_states)
            self.pca_models[layer_key] = pca
            
            benign_transformed = pca.transform(benign_states)
            adversarial_transformed = pca.transform(adversarial_states)
            
            self.benign_centers[layer_key] = np.mean(benign_transformed, axis=0)
            self.adversarial_centers[layer_key] = np.mean(adversarial_transformed, axis=0)
    
    def extract_distance_features(self, hidden_states_data):
        """
        Extract distance features from hidden states.
        For each layer, compute distances to benign and adversarial centers.
        
        Args:
            hidden_states_data: Dictionary containing hidden states
        
        Returns:
            features: Distance feature matrix [n_samples, n_layers * 2]
            labels: Binary labels (0=benign, 1=adversarial)
        """
        all_features = []
        all_labels = []
        
        for item in hidden_states_data['benign']:
            features = []
            for layer_idx in self.layer_indices:
                layer_key = f'layer_{layer_idx}'
                
                hidden_state = item['hidden_states'][layer_key].reshape(1, -1)
                transformed = self.pca_models[layer_key].transform(hidden_state)[0]
                
                dist_benign = np.linalg.norm(transformed - self.benign_centers[layer_key])
                dist_adversarial = np.linalg.norm(transformed - self.adversarial_centers[layer_key])
                
                features.extend([dist_benign, dist_adversarial])
            
            all_features.append(features)
            all_labels.append(0)
        
        for item in hidden_states_data['adversarial']:
            features = []
            for layer_idx in self.layer_indices:
                layer_key = f'layer_{layer_idx}'
                
                hidden_state = item['hidden_states'][layer_key].reshape(1, -1)
                transformed = self.pca_models[layer_key].transform(hidden_state)[0]
                
                dist_benign = np.linalg.norm(transformed - self.benign_centers[layer_key])
                dist_adversarial = np.linalg.norm(transformed - self.adversarial_centers[layer_key])
                
                features.extend([dist_benign, dist_adversarial])
            
            all_features.append(features)
            all_labels.append(1)
        
        return np.array(all_features), np.array(all_labels)

# %%
def generate_layer_combinations(num_layers):
    """
    Generate layer combinations:
    1. All layers
    2. Combinations of early/middle/late layers
    
    Args:
        num_layers: Total number of layers
    
    Returns:
        combinations_dict: Dictionary of layer combinations
    """
    combinations_dict = {}
    
    combinations_dict['all_layers'] = list(range(num_layers))
    
    early_layers = list(range(num_layers // 3))
    middle_start = num_layers // 3
    middle_end = 2 * num_layers // 3
    middle_layers = list(range(middle_start, middle_end))
    late_layers = list(range(2 * num_layers // 3, num_layers))
    
    for e_layer in early_layers[:5]:
        for m_layer in middle_layers[:5]:
            for l_layer in late_layers[:5]:
                comb_name = f'layers_{e_layer}_{m_layer}_{l_layer}'
                combinations_dict[comb_name] = [e_layer, m_layer, l_layer]
    
    return combinations_dict

# %%
layer_combinations = generate_layer_combinations(mt.num_layers)
print(f"Generated {len(layer_combinations)} layer combinations")

# %%
def run_pca_experiments(dataset_names):
    """
    Run PCA-based classification experiments across all dataset combinations.
    
    Args:
        dataset_names: List of dataset names
    
    Returns:
        all_results: Dictionary containing all experiment results
    """
    all_results = {}
    
    for train_idx, train_name in enumerate(dataset_names):
        print(f"\n{'='*60}")
        print(f"Training on: {train_name}")
        print(f"{'='*60}")
        
        train_path = os.path.join(save_dir, f"{train_name}.pkl")
        train_data_full = load_hidden_states(train_path)
        
        benign_count = len(train_data_full['benign'])
        adversarial_count = len(train_data_full['adversarial'])
        
        split_idx_benign = benign_count // 2
        split_idx_adversarial = adversarial_count // 2
        
        train_data = {
            'benign': train_data_full['benign'][:split_idx_benign],
            'adversarial': train_data_full['adversarial'][:split_idx_adversarial]
        }
        
        test_data_same = {
            'benign': train_data_full['benign'][split_idx_benign:],
            'adversarial': train_data_full['adversarial'][split_idx_adversarial:]
        }
        
        train_temp_path = os.path.join(save_dir, f"{train_name}_train_temp.pkl")
        test_same_temp_path = os.path.join(save_dir, f"{train_name}_test_temp.pkl")
        
        with open(train_temp_path, 'wb') as f:
            pickle.dump(train_data, f)
        with open(test_same_temp_path, 'wb') as f:
            pickle.dump(test_data_same, f)
        
        print(f"Training samples: {len(train_data['benign']) + len(train_data['adversarial'])}")
        print(f"Test samples (same dataset): {len(test_data_same['benign']) + len(test_data_same['adversarial'])}")
        
        all_results[train_name] = {}
        
        for comb_name, layer_indices in layer_combinations.items():
            print(f"\n  Layer combination: {comb_name}")
            
            pca_classifier = PCADistanceClassifier(n_components=50, layer_indices=layer_indices)
            pca_classifier.fit(train_data, train_name)
            
            X_train, y_train = pca_classifier.extract_distance_features(train_data)
            
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
            mlp.fit(X_train, y_train)
            
            test_results = {}
            for test_idx, test_name in enumerate(dataset_names):
                if test_idx == train_idx:
                    test_path = test_same_temp_path
                else:
                    test_path = os.path.join(save_dir, f"{test_name}.pkl")
                
                test_data = load_hidden_states(test_path)
                X_test, y_test = pca_classifier.extract_distance_features(test_data)
                y_pred = mlp.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='binary')
                
                test_results[test_name] = {'accuracy': accuracy, 'f1': f1}
                marker = " (SAME)" if train_name == test_name else ""
                print(f"    {test_name:40s}{marker:8s} - Acc: {accuracy:.4f}, F1: {f1:.4f}")
            
            all_results[train_name][comb_name] = {
                'layer_indices': layer_indices,
                'pca_info': pca_classifier.training_info.copy(),
                'test_results': test_results
            }
    
    return all_results

# %%
print("\n" + "="*60)
print("STEP 2: Training and Evaluating on Jailbreak Datasets")
print("="*60 + "\n")

jailbreak_names = [os.path.basename(p).replace('.json', '') for p in jailbreaks]
jailbreak_results = run_pca_experiments(jailbreak_names)

# %%
print("\n" + "="*60)
print("STEP 3: Training and Evaluating on Fairness Datasets")
print("="*60 + "\n")

fairness_names = [os.path.basename(p).replace('.json', '') for p in fairness]
fairness_results = run_pca_experiments(fairness_names)

# %%
print("\n" + "="*60)
print("STEP 4: Training and Evaluating on TruthfulQA Dataset")
print("="*60 + "\n")

truthfulqa_name = os.path.basename(truthfulqa[0]).replace('.json', '')
print(f"Training on: {truthfulqa_name}")

truthfulqa_path = os.path.join(save_dir, f"{truthfulqa_name}.pkl")
truthfulqa_data_full = load_hidden_states(truthfulqa_path)

benign_count = len(truthfulqa_data_full['benign'])
adversarial_count = len(truthfulqa_data_full['adversarial'])

split_idx_benign = benign_count // 2
split_idx_adversarial = adversarial_count // 2

train_data = {
    'benign': truthfulqa_data_full['benign'][:split_idx_benign],
    'adversarial': truthfulqa_data_full['adversarial'][:split_idx_adversarial]
}

test_data = {
    'benign': truthfulqa_data_full['benign'][split_idx_benign:],
    'adversarial': truthfulqa_data_full['adversarial'][split_idx_adversarial:]
}

print(f"Training samples: {len(train_data['benign']) + len(train_data['adversarial'])}")
print(f"Test samples: {len(test_data['benign']) + len(test_data['adversarial'])}")

truthfulqa_results = {}

for comb_name, layer_indices in layer_combinations.items():
    print(f"\nLayer combination: {comb_name}")
    
    pca_classifier = PCADistanceClassifier(n_components=50, layer_indices=layer_indices)
    pca_classifier.fit(train_data, truthfulqa_name)
    
    X_train, y_train = pca_classifier.extract_distance_features(train_data)
    X_test, y_test = pca_classifier.extract_distance_features(test_data)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    truthfulqa_results[comb_name] = {
        'layer_indices': layer_indices,
        'pca_info': pca_classifier.training_info.copy(),
        'accuracy': accuracy,
        'f1': f1
    }
    
    print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# %%
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60 + "\n")

print("JAILBREAK RESULTS:")
print("-" * 60)
for train_name, comb_results in jailbreak_results.items():
    print(f"\nTrained on: {train_name}")
    
    best_comb = max(comb_results.items(), 
                   key=lambda x: np.mean([m['f1'] for m in x[1]['test_results'].values()]))
    comb_name, comb_result = best_comb
    
    print(f"  Best layer combination: {comb_name}")
    for test_name, metrics in comb_result['test_results'].items():
        marker = " (SAME)" if train_name == test_name else ""
        print(f"    {test_name:40s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*60)
print("FAIRNESS RESULTS:")
print("-" * 60)
for train_name, comb_results in fairness_results.items():
    print(f"\nTrained on: {train_name}")
    
    best_comb = max(comb_results.items(), 
                   key=lambda x: np.mean([m['f1'] for m in x[1]['test_results'].values()]))
    comb_name, comb_result = best_comb
    
    print(f"  Best layer combination: {comb_name}")
    for test_name, metrics in comb_result['test_results'].items():
        marker = " (SAME)" if train_name == test_name else ""
        print(f"    {test_name:40s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*60)
print("TRUTHFULQA RESULTS:")
print("-" * 60)

best_tqa = max(truthfulqa_results.items(), key=lambda x: x[1]['f1'])
comb_name, comb_result = best_tqa

print(f"Best layer combination: {comb_name}")
print(f"Accuracy: {comb_result['accuracy']:.4f}, F1: {comb_result['f1']:.4f}")

print("\n" + "="*60)
print("Experiment completed!")
print("="*60)

# %%
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

all_results = {
    'jailbreak': {},
    'fairness': {},
    'truthfulqa': {}
}

for train_name, comb_results in jailbreak_results.items():
    all_results['jailbreak'][train_name] = {}
    for comb_name, result in comb_results.items():
        all_results['jailbreak'][train_name][comb_name] = {
            'layer_indices': result['layer_indices'],
            'pca_info': result['pca_info'],
            'test_results': {k: {'accuracy': float(v['accuracy']), 'f1': float(v['f1'])} 
                           for k, v in result['test_results'].items()}
        }

for train_name, comb_results in fairness_results.items():
    all_results['fairness'][train_name] = {}
    for comb_name, result in comb_results.items():
        all_results['fairness'][train_name][comb_name] = {
            'layer_indices': result['layer_indices'],
            'pca_info': result['pca_info'],
            'test_results': {k: {'accuracy': float(v['accuracy']), 'f1': float(v['f1'])} 
                           for k, v in result['test_results'].items()}
        }

for comb_name, result in truthfulqa_results.items():
    all_results['truthfulqa'][comb_name] = {
        'layer_indices': result['layer_indices'],
        'pca_info': result['pca_info'],
        'accuracy': float(result['accuracy']),
        'f1': float(result['f1'])
    }

results_file = os.path.join(results_dir, 'llama2_pca_distance_classification_results.json')
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to: {results_file}")