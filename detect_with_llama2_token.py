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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from scipy import stats
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
print("Test Generate output",generate_outputs(test_prompt, mt))

# %%
def analyze_token_level_ace(prompt, mt, num_replacements=200, random_seed=42):
    """
    Analyze ACE (Activation Change Effect) at token level by replacing individual tokens
    with "!" and measuring the impact on model predictions.
    
    Args:
        prompt: Input prompt string
        mt: ModelAndTokenizer object
        num_replacements: Number of random token replacements if sequence length > 40
        random_seed: Random seed for reproducibility
    
    Returns:
        dict containing:
            - original_logit: Original probability for predicted token
            - ace_values: List of ACE values (absolute probability changes) for each position
            - statistics: Dict with mean, std, range, skewness, kurtosis of ACE values
    """
    random.seed(random_seed)
    
    # Prepare input
    inp = make_inputs(mt.tokenizer, [prompt])
    
    # Get original prediction
    with torch.no_grad():
        answer_t, original_logit = [d[0] for d in predict_from_input(mt.model, inp)]
    
    # Get input tokens
    input_ids = inp['input_ids'][0]
    seq_length = len(input_ids)
    
    # Get token ID for "!" replacement
    replacement_token_id = mt.tokenizer.encode("!", add_special_tokens=False)[0]
    
    # Determine which positions to analyze (skip first 3 and last 3 tokens)
    analyzable_positions = list(range(3, seq_length - 3))
    
    if len(analyzable_positions) > num_replacements:
        positions_to_analyze = random.sample(analyzable_positions, num_replacements)
    else:
        positions_to_analyze = analyzable_positions
    
    ace_values = []
    
    # Calculate ACE for each position
    for pos in positions_to_analyze:
        # Create a copy of input with token replaced by "!"
        inp_patched = {k: v.clone() if torch.is_tensor(v) else v for k, v in inp.items()}
        inp_patched['input_ids'][0, pos] = replacement_token_id
        
        # Get probability after replacement
        with torch.no_grad():
            outputs = mt.model(**inp_patched)
        
        replaced_prob = torch.softmax(outputs.logits[0, -1, :], dim=0)[answer_t]
        
        # Calculate ACE: absolute change in probability
        ace_value = abs(original_logit.item() - replaced_prob.item())
        ace_values.append(ace_value)
    
    # Calculate statistics
    ace_array = np.array(ace_values)
    
    statistics = {
        'mean': np.mean(ace_array),
        'std': np.std(ace_array, ddof=1),
        'range': np.max(ace_array) - np.min(ace_array),
        'skewness': stats.skew(ace_array),
        'kurtosis': stats.kurtosis(ace_array, fisher=False)
    }
    
    return {
        'original_logit': original_logit.item(),
        'ace_values': ace_values,
        'statistics': statistics
    }

# %%
# Test the function
result = analyze_token_level_ace(test_prompt, mt)
print("Statistics:")
for key, value in result['statistics'].items():
    print(f"  {key}: {value:.6f}")

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
save_dir = "./datasets/token"
os.makedirs(save_dir, exist_ok=True)

# %%
def extract_and_save_features(data_path, save_name):
    """
    Extract ACE features from dataset and save to JSON file.
    Features include: original_logit + 5 statistics (mean, std, range, skewness, kurtosis).
    
    Args:
        data_path: Path to input data file
        save_name: Name for saved feature file
    """
    save_path = os.path.join(save_dir, f"{save_name}.json")
    
    # Skip if already exists
    if os.path.exists(save_path):
        print(f"Features already exist for {save_name}, skipping...")
        return
    
    print(f"Processing {save_name}...")
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Prepare results dictionary
    results = {
        'benign': [],
        'adversarial': []
    }
    
    # Process benign samples
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
            result = analyze_token_level_ace(formatted_prompt, mt)
            
            # Extract features: original_logit + statistics
            features = [result['original_logit']]
            features.extend([
                result['statistics']['mean'],
                result['statistics']['std'],
                result['statistics']['range'],
                result['statistics']['skewness'],
                result['statistics']['kurtosis']
            ])
            
            results['benign'].append({
                'prompt': prompt,
                'features': [float(x) for x in features],
                'result': result
            })
        except Exception as e:
            print(f"Error processing benign sample: {e}")
            continue
    
    # Process adversarial samples
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
            result = analyze_token_level_ace(formatted_prompt, mt)
            
            # Extract features: original_logit + statistics
            features = [result['original_logit']]
            features.extend([
                result['statistics']['mean'],
                result['statistics']['std'],
                result['statistics']['range'],
                result['statistics']['skewness'],
                result['statistics']['kurtosis']
            ])
            
            results['adversarial'].append({
                'prompt': prompt,
                'features': [float(x) for x in features],
                'result': result
            })
        except Exception as e:
            print(f"Error processing adversarial sample: {e}")
            continue
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved features to {save_path}")
    print(f"Benign samples: {len(results['benign'])}, Adversarial samples: {len(results['adversarial'])}\n")

# %%
# Extract and save features for all datasets
print("="*60)
print("STEP 1: Extracting and saving features for all datasets")
print("="*60 + "\n")

# Jailbreak datasets
for jb_path in jailbreaks:
    dataset_name = os.path.basename(jb_path).replace('.json', '')
    extract_and_save_features(jb_path, dataset_name)

# Fairness datasets
for fair_path in fairness:
    dataset_name = os.path.basename(fair_path).replace('.json', '')
    extract_and_save_features(fair_path, dataset_name)

# TruthfulQA dataset
for tqa_path in truthfulqa:
    dataset_name = os.path.basename(tqa_path).replace('.json', '')
    extract_and_save_features(tqa_path, dataset_name)

print("Feature extraction completed!\n")

# %%
def load_features_from_json(json_path):
    """
    Load features from JSON file.
    
    Returns:
        X: Feature matrix (original_logit + 5 statistics)
        y: Labels (0=benign, 1=adversarial)
        prompts: Original prompts
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    prompts = []
    
    # Load benign data
    for item in data['benign']:
        X.append(item['features'])
        y.append(0)
        prompts.append(item['prompt'])
    
    # Load adversarial data
    for item in data['adversarial']:
        X.append(item['features'])
        y.append(1)
        prompts.append(item['prompt'])
    
    return np.array(X), np.array(y), prompts

# %%
def train_and_evaluate_mlp(X_train, y_train, X_test, y_test):
    """
    Train and evaluate MLP classifier with (128, 64) hidden layers.
    
    Returns:
        clf: Trained classifier
        accuracy: Test accuracy
        f1: Test F1 score
        y_pred: Predictions
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    return clf, accuracy, f1, y_pred

# %%
# Train and evaluate on Jailbreak datasets
print("="*60)
print("STEP 2: Training and Evaluating on Jailbreak Datasets")
print("="*60 + "\n")

jailbreak_names = [os.path.basename(p).replace('.json', '') for p in jailbreaks]
jailbreak_results = {}

for train_idx, train_name in enumerate(jailbreak_names):
    print(f"\n{'='*60}")
    print(f"Training on: {train_name}")
    print(f"{'='*60}")
    
    # Load training data
    train_path = os.path.join(save_dir, f"{train_name}.json")
    X_full, y_full, prompts_full = load_features_from_json(train_path)
    
    # Split training set (50% train, 50% test on same dataset)
    X_train, X_test_same, y_train, y_test_same = train_test_split(
        X_full, y_full, train_size=0.5, random_state=42, stratify=y_full
    )
    
    print(f"Training samples: {len(X_train)}, Test samples (same dataset): {len(X_test_same)}")
    print(f"Feature dimension: {X_train.shape[1]} (original_logit + 5 statistics)")
    
    # Train model
    clf, _, _, _ = train_and_evaluate_mlp(X_train, y_train, X_test_same, y_test_same)
    
    # Test on all jailbreak datasets
    results = {}
    for test_idx, test_name in enumerate(jailbreak_names):
        test_path = os.path.join(save_dir, f"{test_name}.json")
        
        if test_idx == train_idx:
            # Same dataset, use previously split test set
            X_test = X_test_same
            y_test = y_test_same
        else:
            # Different dataset, load full data
            X_test, y_test, _ = load_features_from_json(test_path)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        results[test_name] = {'accuracy': accuracy, 'f1': f1}
        print(f"  Test on {test_name:30s} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    jailbreak_results[train_name] = results

# %%
# Train and evaluate on Fairness datasets
print("\n" + "="*60)
print("STEP 3: Training and Evaluating on Fairness Datasets")
print("="*60 + "\n")

fairness_names = [os.path.basename(p).replace('.json', '') for p in fairness]
fairness_results = {}

for train_idx, train_name in enumerate(fairness_names):
    print(f"\n{'='*60}")
    print(f"Training on: {train_name}")
    print(f"{'='*60}")
    
    # Load training data
    train_path = os.path.join(save_dir, f"{train_name}.json")
    X_full, y_full, prompts_full = load_features_from_json(train_path)
    
    # Split training set
    X_train, X_test_same, y_train, y_test_same = train_test_split(
        X_full, y_full, train_size=0.5, random_state=42, stratify=y_full
    )
    
    print(f"Training samples: {len(X_train)}, Test samples (same dataset): {len(X_test_same)}")
    print(f"Feature dimension: {X_train.shape[1]} (original_logit + 5 statistics)")
    
    # Train model
    clf, _, _, _ = train_and_evaluate_mlp(X_train, y_train, X_test_same, y_test_same)
    
    # Test on all fairness datasets
    results = {}
    for test_idx, test_name in enumerate(fairness_names):
        test_path = os.path.join(save_dir, f"{test_name}.json")
        
        if test_idx == train_idx:
            X_test = X_test_same
            y_test = y_test_same
        else:
            X_test, y_test, _ = load_features_from_json(test_path)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        results[test_name] = {'accuracy': accuracy, 'f1': f1}
        print(f"  Test on {test_name:40s} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    fairness_results[train_name] = results

# %%
# Train and evaluate on TruthfulQA dataset
print("\n" + "="*60)
print("STEP 4: Training and Evaluating on TruthfulQA Dataset")
print("="*60 + "\n")

truthfulqa_name = os.path.basename(truthfulqa[0]).replace('.json', '')
print(f"Training on: {truthfulqa_name}")

# Load TruthfulQA data
truthfulqa_path = os.path.join(save_dir, f"{truthfulqa_name}.json")
X_full, y_full, _ = load_features_from_json(truthfulqa_path)

# Split training set (50% train, 50% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, train_size=0.5, random_state=42, stratify=y_full
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Feature dimension: {X_train.shape[1]} (original_logit + 5 statistics)")

# Train and test
clf, accuracy, f1, _ = train_and_evaluate_mlp(X_train, y_train, X_test, y_test)
print(f"  Test on {truthfulqa_name:30s} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# %%
# Print final summary
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
# Save all results to JSON file
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

all_results = {
    'jailbreak': {},
    'fairness': {},
    'truthfulqa': {}
}

# Organize jailbreak results
for train_name, test_results in jailbreak_results.items():
    all_results['jailbreak'][train_name] = {
        'test_results': {}
    }
    
    for test_name, metrics in test_results.items():
        all_results['jailbreak'][train_name]['test_results'][test_name] = {
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1'])
        }

# Organize fairness results
for train_name, test_results in fairness_results.items():
    all_results['fairness'][train_name] = {
        'test_results': {}
    }
    
    for test_name, metrics in test_results.items():
        all_results['fairness'][train_name]['test_results'][test_name] = {
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1'])
        }

# Organize truthfulqa results
all_results['truthfulqa'] = {
    'accuracy': float(accuracy),
    'f1': float(f1)
}

# Save to JSON file
results_file = os.path.join(results_dir, 'llama2_token_level_ace_results.json')
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to: {results_file}")
print(f"\nResults structure:")
print(f"  - Jailbreak datasets: {len(all_results['jailbreak'])} training sets")
print(f"  - Fairness datasets: {len(all_results['fairness'])} training sets")
print(f"  - TruthfulQA: 1 dataset")