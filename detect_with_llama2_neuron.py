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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
import random

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
save_dir = "./datasets/neuron"
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
class NeuronLevelClassifier:
    """
    Two-stage neuron-level classifier:
    Stage 1: Train Logistic Regression on each layer's hidden states (neurons)
    Stage 2: Use LR prediction probabilities as features for MLP classifier
    """
    def __init__(self, num_layers):
        """
        Args:
            num_layers: Total number of layers in the model
        """
        self.num_layers = num_layers
        self.lr_models = {}
        self.mlp_model = None
    
    def train_layer_lr(self, hidden_states_data):
        """
        Train Logistic Regression model for each layer.
        
        Args:
            hidden_states_data: Dictionary containing benign and adversarial hidden states
        """
        print("Training Logistic Regression models for each layer...")
        
        for layer_idx in tqdm(range(self.num_layers), desc="Training LR models"):
            layer_key = f'layer_{layer_idx}'
            
            X_train = []
            y_train = []
            
            for item in hidden_states_data['benign']:
                X_train.append(item['hidden_states'][layer_key])
                y_train.append(0)
            
            for item in hidden_states_data['adversarial']:
                X_train.append(item['hidden_states'][layer_key])
                y_train.append(1)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            lr = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs',
                n_jobs=-1
            )
            lr.fit(X_train, y_train)
            
            self.lr_models[layer_key] = lr
    
    def extract_lr_predictions(self, hidden_states_data):
        """
        Extract prediction probabilities from trained LR models as features.
        
        Args:
            hidden_states_data: Dictionary containing hidden states
        
        Returns:
            X: Feature matrix [n_samples, n_layers] of LR prediction probabilities
            y: Labels (0=benign, 1=adversarial)
        """
        all_features = []
        all_labels = []
        
        for item in hidden_states_data['benign']:
            features = []
            for layer_idx in range(self.num_layers):
                layer_key = f'layer_{layer_idx}'
                hidden_state = item['hidden_states'][layer_key].reshape(1, -1)
                prob = self.lr_models[layer_key].predict_proba(hidden_state)[0, 1]
                features.append(prob)
            
            all_features.append(features)
            all_labels.append(0)
        
        for item in hidden_states_data['adversarial']:
            features = []
            for layer_idx in range(self.num_layers):
                layer_key = f'layer_{layer_idx}'
                hidden_state = item['hidden_states'][layer_key].reshape(1, -1)
                prob = self.lr_models[layer_key].predict_proba(hidden_state)[0, 1]
                features.append(prob)
            
            all_features.append(features)
            all_labels.append(1)
        
        return np.array(all_features), np.array(all_labels)
    
    def train_mlp(self, X_train, y_train):
        """
        Train MLP classifier with (128, 64) hidden layers.
        
        Args:
            X_train: LR prediction probability features
            y_train: Labels
        """
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
    
    def predict(self, hidden_states_data):
        """
        Make predictions on new data.
        
        Args:
            hidden_states_data: Dictionary containing hidden states
        
        Returns:
            y_pred: Predictions
        """
        features, _ = self.extract_lr_predictions(hidden_states_data)
        return self.mlp_model.predict(features)

# %%
def train_and_evaluate_neuron_classifier(train_path, test_path):
    """
    Train neuron-level classifier and evaluate on test set.
    
    Args:
        train_path: Path to training data pickle file
        test_path: Path to test data pickle file
    
    Returns:
        accuracy: Test accuracy
        f1: Test F1 score
    """
    train_data = load_hidden_states(train_path)
    test_data = load_hidden_states(test_path)
    
    classifier = NeuronLevelClassifier(mt.num_layers)
    
    classifier.train_layer_lr(train_data)
    
    X_train, y_train = classifier.extract_lr_predictions(train_data)
    classifier.train_mlp(X_train, y_train)
    
    X_test, y_test = classifier.extract_lr_predictions(test_data)
    y_pred = classifier.mlp_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    return accuracy, f1

# %%
print("="*60)
print("STEP 2: Training and Evaluating on Jailbreak Datasets")
print("="*60 + "\n")

jailbreak_names = [os.path.basename(p).replace('.json', '') for p in jailbreaks]
jailbreak_results = {}

for train_idx, train_name in enumerate(jailbreak_names):
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
    
    classifier = NeuronLevelClassifier(mt.num_layers)
    classifier.train_layer_lr(train_data)
    
    X_train, y_train = classifier.extract_lr_predictions(train_data)
    classifier.train_mlp(X_train, y_train)
    
    results = {}
    for test_idx, test_name in enumerate(jailbreak_names):
        if test_idx == train_idx:
            test_path = test_same_temp_path
        else:
            test_path = os.path.join(save_dir, f"{test_name}.pkl")
        
        test_data = load_hidden_states(test_path)
        X_test, y_test = classifier.extract_lr_predictions(test_data)
        y_pred = classifier.mlp_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        results[test_name] = {'accuracy': accuracy, 'f1': f1}
        print(f"  Test on {test_name:30s} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    jailbreak_results[train_name] = results

# %%
print("\n" + "="*60)
print("STEP 3: Training and Evaluating on Fairness Datasets")
print("="*60 + "\n")

fairness_names = [os.path.basename(p).replace('.json', '') for p in fairness]
fairness_results = {}

for train_idx, train_name in enumerate(fairness_names):
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
    
    classifier = NeuronLevelClassifier(mt.num_layers)
    classifier.train_layer_lr(train_data)
    
    X_train, y_train = classifier.extract_lr_predictions(train_data)
    classifier.train_mlp(X_train, y_train)
    
    results = {}
    for test_idx, test_name in enumerate(fairness_names):
        if test_idx == train_idx:
            test_path = test_same_temp_path
        else:
            test_path = os.path.join(save_dir, f"{test_name}.pkl")
        
        test_data = load_hidden_states(test_path)
        X_test, y_test = classifier.extract_lr_predictions(test_data)
        y_pred = classifier.mlp_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        results[test_name] = {'accuracy': accuracy, 'f1': f1}
        print(f"  Test on {test_name:40s} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    fairness_results[train_name] = results

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

classifier = NeuronLevelClassifier(mt.num_layers)
classifier.train_layer_lr(train_data)

X_train, y_train = classifier.extract_lr_predictions(train_data)
classifier.train_mlp(X_train, y_train)

X_test, y_test = classifier.extract_lr_predictions(test_data)
y_pred = classifier.mlp_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')

print(f"  Test on {truthfulqa_name:30s} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# %%
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60 + "\n")

print("JAILBREAK RESULTS:")
print("-" * 60)
for train_name, results in jailbreak_results.items():
    print(f"\nTrained on: {train_name}")
    for test_name, metrics in results.items():
        marker = " (SAME)" if train_name == test_name else ""
        print(f"  {test_name:30s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*60)
print("FAIRNESS RESULTS:")
print("-" * 60)
for train_name, results in fairness_results.items():
    print(f"\nTrained on: {train_name}")
    for test_name, metrics in results.items():
        marker = " (SAME)" if train_name == test_name else ""
        print(f"  {test_name:40s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*60)
print("TRUTHFULQA RESULTS:")
print("-" * 60)
print(f"Trained and tested on: {truthfulqa_name}")
print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

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

for train_name, test_results in jailbreak_results.items():
    all_results['jailbreak'][train_name] = {
        'test_results': {}
    }
    
    for test_name, metrics in test_results.items():
        all_results['jailbreak'][train_name]['test_results'][test_name] = {
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1'])
        }

for train_name, test_results in fairness_results.items():
    all_results['fairness'][train_name] = {
        'test_results': {}
    }
    
    for test_name, metrics in test_results.items():
        all_results['fairness'][train_name]['test_results'][test_name] = {
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1'])
        }

all_results['truthfulqa'] = {
    'accuracy': float(accuracy),
    'f1': float(f1)
}

results_file = os.path.join(results_dir, 'llama2_neuron_level_classification_results.json')
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to: {results_file}")
print(f"\nResults structure:")
print(f"  - Jailbreak datasets: {len(all_results['jailbreak'])} training sets")
print(f"  - Fairness datasets: {len(all_results['fairness'])} training sets")
print(f"  - TruthfulQA: 1 dataset")