import csv
import os
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd

def read_token_lang_frequencies(file_path):
    """Read token to language mapping from CSV file."""
    token_lang_frequencies = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            token_lang_frequencies[row[0]] = row[1]  
    return token_lang_frequencies

def read_layer_predictions(file_path):
    """Read layer predictions from experiment results CSV."""
    layer_predictions = defaultdict(lambda: defaultdict(list))
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        current_prompt = None
        prompt_id = -1
        for row in reader:
            prompt_text = row['Prompt']
            layer = int(row['Layer'])
            
            # Parse predictions from format: "token (prob), token (prob), ..."
            predictions = []
            for pred in row['Top Predictions'].split(','):
                parts = pred.strip().rsplit(' ', 1)
                if len(parts) == 2:
                    tok, prob = parts[0], parts[1].replace('(', '').replace(')', '')
                    try:
                        predictions.append((tok, float(prob)))
                    except ValueError:
                        print(f"Skipping malformed prediction: {pred}")
            
            if prompt_text != current_prompt:
                prompt_id += 1
                current_prompt = prompt_text
            
            layer_predictions[prompt_id][layer] = predictions
    
    return layer_predictions

def get_language_frequencies(layer_predictions, token_lang_frequencies, lang_list: List[str]):
    """Compute aggregated language frequencies per layer."""
    total_lang_frequencies = defaultdict(lambda: defaultdict(float))
    for prompt_id, layers in layer_predictions.items():
        for layer, predictions in layers.items():
            for token, prob in predictions:
                lang = token_lang_frequencies.get(token, None)
                if lang in lang_list:
                    total_lang_frequencies[layer][lang] += prob
    return total_lang_frequencies

def print_language_frequencies(total_lang_frequencies):
    """Print language frequencies for each layer."""
    for layer in sorted(total_lang_frequencies.keys()):
        lang_freq = total_lang_frequencies[layer]
        print(f"Layer {layer}: {dict(lang_freq)}")

def save_language_frequencies(total_lang_frequencies, output_path):
    """Save language frequencies to CSV file."""
    rows = []
    for layer in sorted(total_lang_frequencies.keys()):
        lang_freq = total_lang_frequencies[layer]
        row = {"Layer": layer}
        row.update(lang_freq)
        rows.append(row)
    

    all_langs = set()
    for lang_freq in total_lang_frequencies.values():
        all_langs.update(lang_freq.keys())
    
    fieldnames = ["Layer"] + sorted(all_langs)
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Results saved to {output_path}")

def process_all_results(
    results_dir: str,
    token_lang_freq_file: str,
    output_dir: str,
    lang_list: List[str]
) -> None:
    """
    Process all experiment result CSV files in a directory.
    
    Args:
        results_dir: Directory containing prediction CSV files
        token_lang_freq_file: Path to token-language frequency mapping file
        output_dir: Directory to save processed results
        lang_list: List of languages to track
    """
    # Load token-language mapping
    print("Loading token-language frequencies...")
    token_lang_frequencies = read_token_lang_frequencies(token_lang_freq_file)

    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    print(f"\nProcessing {len(csv_files)} result files...\n")
    
    for csv_file in sorted(csv_files):
        input_path = os.path.join(results_dir, csv_file)
        output_filename = csv_file.replace('.csv', '_language_frequencies.csv')
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing: {csv_file}")
        
        try:
            # Read predictions
            layer_predictions = read_layer_predictions(input_path)
            
            # Compute language frequencies
            total_lang_frequencies = get_language_frequencies(
                layer_predictions,
                token_lang_frequencies,
                lang_list
            )
            
            # Print 
            print_language_frequencies(total_lang_frequencies)
            
            # Save to file
            save_language_frequencies(total_lang_frequencies, output_path)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}\n")
        
        print()
