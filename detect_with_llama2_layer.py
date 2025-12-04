# %%
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
generate_outputs(test_prompt, mt)

# %%
def trace_with_patch_layer(model, inp, states_to_patch, answers_t):
    """
    Trace model execution with layer patching by creating shortcuts between layers.
    
    Args:
        model: The language model
        inp: Input tensors
        states_to_patch: List of two layer names [source_layer, target_layer]
        answers_t: Answer token IDs to track probabilities
    
    Returns:
        probs: Probability of the answer token after patching
    """
    prng = np.random.RandomState(1)
    layers = [states_to_patch[0], states_to_patch[1]]

    inter_results = {}

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def patch_rep(x, layer):
        """
        Patching function that creates a shortcut connection.
        Stores hidden states from first layer and replaces second layer input.
        """
        if layer not in layers:
            return x

        if layer == layers[0]:
            inter_results["hidden_states"] = x[0].cpu()
            return x
        elif layer == layers[1]:
            short_cut_1 = inter_results["hidden_states"].cuda()
            short_cut = (short_cut_1,)
            return short_cut
            
    with torch.no_grad(), nethook.TraceDict(
        model,
        layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs

# %%
def analyse_based_on_single_layer(prompt):
    """
    Analyze ACE by creating shortcuts between consecutive layers (layer i -> layer i+1).
    
    Args:
        prompt: Input prompt string
    
    Returns:
        logits: Original prediction probability
        data_on_cpu: List of ACE values (absolute probability changes) for each layer
    """
    inp = make_inputs(mt.tokenizer, [prompt]*2)
    with torch.no_grad():
        answer_t, logits = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    
    model = mt.model
    result_prob = []
    for layer in range(mt.num_layers-1):
        layers = [layername(model, layer), layername(model, layer + 1)]
        prob = trace_with_patch_layer(model, inp, layers, answer_t)
        result_prob.append(prob)
    
    data_on_cpu = [abs(x.item() - logits.item()) for x in result_prob]
        
    return logits.item(), data_on_cpu

# %%
analyse_based_on_single_layer(test_prompt)

# %%
def analyse_based_on_multi_layer(prompt):
    """
    Analyze ACE by creating shortcuts spanning 3 layers (layer i -> layer i+3).
    
    Args:
        prompt: Input prompt string
    
    Returns:
        logits: Original prediction probability
        data_on_cpu: List of ACE values (absolute probability changes) for each layer
    """
    inp = make_inputs(mt.tokenizer, [prompt]*2)
    with torch.no_grad():
        answer_t, logits = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    
    model = mt.model
    result_prob = []
    for layer in range(mt.num_layers-3):
        layers = [layername(model, layer), layername(model, layer + 3)]
        prob = trace_with_patch_layer(model, inp, layers, answer_t)
        result_prob.append(prob)
    
    data_on_cpu = [abs(x.item() - logits.item()) for x in result_prob]
        
    return logits.item(), data_on_cpu

# %%
analyse_based_on_multi_layer(test_prompt)

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
save_dir = "./datasets/layers"
os.makedirs(save_dir, exist_ok=True)

# %%
def extract_and_save_features(data_path, save_name):
    """
    Extract layer-based ACE features from dataset and save to JSON file.
    Features include single-layer shortcuts, multi-layer shortcuts, and original logits.
    
    Args:
        data_path: Path to input data file
        save_name: Name for saved feature file
    """
    save_path = os.path.join(save_dir, f"{save_name}.json")
    
    if os.path.exists(save_path):
        print(f"Features already exist for {save_name}, skipping...")
        return
    
    print(f"Processing {save_name}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
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
            logits_single, ace_single = analyse_based_on_single_layer(formatted_prompt)
            logits_multi, ace_multi = analyse_based_on_multi_layer(formatted_prompt)
            
            results['benign'].append({
                'prompt': prompt,
                'logits_single': float(logits_single),
                'logits_multi': float(logits_multi),
                'ace_single': [float(x) for x in ace_single],
                'ace_multi': [float(x) for x in ace_multi]
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
            logits_single, ace_single = analyse_based_on_single_layer(formatted_prompt)
            logits_multi, ace_multi = analyse_based_on_multi_layer(formatted_prompt)
            
            results['adversarial'].append({
                'prompt': prompt,
                'logits_single': float(logits_single),
                'logits_multi': float(logits_multi),
                'ace_single': [float(x) for x in ace_single],
                'ace_multi': [float(x) for x in ace_multi]
            })
        except Exception as e:
            print(f"Error processing adversarial sample: {e}")
            continue
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved features to {save_path}")
    print(f"Benign samples: {len(results['benign'])}, Adversarial samples: {len(results['adversarial'])}\n")

# %%
# Extract and save features for all datasets
print("="*70)
print("STEP 1: Extracting and saving features for all datasets")
print("="*70 + "\n")

for jb_path in jailbreaks:
    dataset_name = os.path.basename(jb_path).replace('.json', '')
    extract_and_save_features(jb_path, dataset_name)

for fair_path in fairness:
    dataset_name = os.path.basename(fair_path).replace('.json', '')
    extract_and_save_features(fair_path, dataset_name)

for tqa_path in truthfulqa:
    dataset_name = os.path.basename(tqa_path).replace('.json', '')
    extract_and_save_features(tqa_path, dataset_name)

print("Feature extraction completed!\n")

# %%
def load_features_from_json(json_path, feature_type='combined'):
    """
    Load features from JSON file.
    
    Args:
        json_path: Path to JSON file
        feature_type: 'single' (consecutive layers), 'multi' (3-layer span), or 'combined'
    
    Returns:
        X: Feature matrix
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
        if feature_type == 'single':
            features = item['ace_single'] + [item['logits_single']]
        elif feature_type == 'multi':
            features = item['ace_multi'] + [item['logits_multi']]
        else:  # combined
            features = item['ace_single'] + item['ace_multi'] + [item['logits_single'], item['logits_multi']]
        
        X.append(features)
        y.append(0)
        prompts.append(item['prompt'])
    
    # Load adversarial data
    for item in data['adversarial']:
        if feature_type == 'single':
            features = item['ace_single'] + [item['logits_single']]
        elif feature_type == 'multi':
            features = item['ace_multi'] + [item['logits_multi']]
        else:  # combined
            features = item['ace_single'] + item['ace_multi'] + [item['logits_single'], item['logits_multi']]
        
        X.append(features)
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
def run_experiments(dataset_names, dataset_type='jailbreak'):
    """
    Run experiments across all feature types for given datasets.
    
    Args:
        dataset_names: List of dataset names
        dataset_type: Type of dataset (for logging)
    
    Returns:
        all_results: Dictionary containing all experiment results
    """
    feature_types = ['single', 'multi', 'combined']
    all_results = {}
    
    for train_idx, train_name in enumerate(dataset_names):
        print(f"\n{'='*70}")
        print(f"Training on: {train_name}")
        print(f"{'='*70}")
        
        train_path = os.path.join(save_dir, f"{train_name}.json")
        
        for feature_type in feature_types:
            print(f"\n  Feature Type: {feature_type.upper()}")
            print(f"  {'-'*66}")
            
            # Load training data
            X_full, y_full, _ = load_features_from_json(train_path, feature_type)
            X_train, X_test_same, y_train, y_test_same = train_test_split(
                X_full, y_full, train_size=0.5, random_state=42, stratify=y_full
            )
            
            print(f"    Training samples: {len(X_train)}, Feature dimension: {X_train.shape[1]}")
            
            # Train model
            clf, _, _, _ = train_and_evaluate_mlp(X_train, y_train, X_test_same, y_test_same)
            
            # Test on all datasets
            results = {}
            for test_idx, test_name in enumerate(dataset_names):
                test_path = os.path.join(save_dir, f"{test_name}.json")
                
                if test_idx == train_idx:
                    X_test = X_test_same
                    y_test = y_test_same
                else:
                    X_test, y_test, _ = load_features_from_json(test_path, feature_type)
                
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='binary')
                
                results[test_name] = {'accuracy': accuracy, 'f1': f1}
                marker = " (SAME)" if train_name == test_name else ""
                print(f"      {test_name:35s}{marker:8s} - Acc: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Save results
            key = f"{train_name}_{feature_type}"
            all_results[key] = {'results': results}
    
    return all_results

# %%
# Get dataset names
jailbreak_names = [os.path.basename(p).replace('.json', '') for p in jailbreaks]
fairness_names = [os.path.basename(p).replace('.json', '') for p in fairness]

# %%
# Run Jailbreak experiments
print("="*70)
print("STEP 2: Training and Evaluating on Jailbreak Datasets")
print("="*70)
jailbreak_results = run_experiments(jailbreak_names, 'jailbreak')

# %%
# Run Fairness experiments
print("\n" + "="*70)
print("STEP 3: Training and Evaluating on Fairness Datasets")
print("="*70)
fairness_results = run_experiments(fairness_names, 'fairness')

# %%
# Run TruthfulQA experiments
print("\n" + "="*70)
print("STEP 4: Training and Evaluating on TruthfulQA Dataset")
print("="*70 + "\n")

truthfulqa_name = os.path.basename(truthfulqa[0]).replace('.json', '')
truthfulqa_path = os.path.join(save_dir, f"{truthfulqa_name}.json")
truthfulqa_results = {}

feature_types = ['single', 'multi', 'combined']
for feature_type in feature_types:
    print(f"\nFeature Type: {feature_type.upper()}")
    print("-" * 70)
    
    X_full, y_full, _ = load_features_from_json(truthfulqa_path, feature_type)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, train_size=0.5, random_state=42, stratify=y_full
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    clf, accuracy, f1, _ = train_and_evaluate_mlp(X_train, y_train, X_test, y_test)
    
    truthfulqa_results[feature_type] = {
        'accuracy': accuracy,
        'f1': f1
    }
    
    print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# %%
# Print final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70 + "\n")

print("JAILBREAK RESULTS:")
print("-" * 70)
for key, value in jailbreak_results.items():
    parts = key.rsplit('_', 1)
    train_name = parts[0]
    feature_type = parts[1]
    print(f"\nTrained on: {train_name} | Feature: {feature_type}")
    for test_name, metrics in value['results'].items():
        marker = " (SAME)" if train_name == test_name else ""
        print(f"  {test_name:35s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*70)
print("FAIRNESS RESULTS:")
print("-" * 70)
for key, value in fairness_results.items():
    parts = key.rsplit('_', 1)
    train_name = parts[0]
    feature_type = parts[1]
    print(f"\nTrained on: {train_name} | Feature: {feature_type}")
    for test_name, metrics in value['results'].items():
        marker = " (SAME)" if train_name == test_name else ""
        print(f"  {test_name:40s}{marker:8s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

print("\n" + "="*70)
print("TRUTHFULQA RESULTS:")
print("-" * 70)
for feature_type, result in truthfulqa_results.items():
    print(f"\nFeature Type: {feature_type.upper()}")
    print(f"  Trained and tested on: {truthfulqa_name}")
    print(f"  Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")

print("\n" + "="*70)
print("Experiment completed!")
print("="*70)

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
for key, value in jailbreak_results.items():
    parts = key.rsplit('_', 1)
    train_name = parts[0]
    feature_type = parts[1]
    
    if train_name not in all_results['jailbreak']:
        all_results['jailbreak'][train_name] = {}
    
    all_results['jailbreak'][train_name][feature_type] = {
        'test_results': {}
    }
    
    for test_name, metrics in value['results'].items():
        all_results['jailbreak'][train_name][feature_type]['test_results'][test_name] = {
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1'])
        }

# Organize fairness results
for key, value in fairness_results.items():
    parts = key.rsplit('_', 1)
    train_name = parts[0]
    feature_type = parts[1]
    
    if train_name not in all_results['fairness']:
        all_results['fairness'][train_name] = {}
    
    all_results['fairness'][train_name][feature_type] = {
        'test_results': {}
    }
    
    for test_name, metrics in value['results'].items():
        all_results['fairness'][train_name][feature_type]['test_results'][test_name] = {
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1'])
        }

# Organize truthfulqa results
truthfulqa_formatted = {}
for feature_type, result in truthfulqa_results.items():
    truthfulqa_formatted[feature_type] = {
        'accuracy': float(result['accuracy']),
        'f1': float(result['f1'])
    }
all_results['truthfulqa'] = truthfulqa_formatted

# Save to JSON file
results_file = os.path.join(results_dir, 'llama2_layer_experiment_results.json')
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to: {results_file}")
print(f"\nResults structure:")
print(f"  - Jailbreak datasets: {len(all_results['jailbreak'])} training sets")
print(f"  - Fairness datasets: {len(all_results['fairness'])} training sets")
print(f"  - TruthfulQA: 1 dataset")