import random
import torch
from typing import List, Tuple
import csv
from torch.cuda.amp import autocast
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def construct_few_shot_prompts(
    source_prompts: List[str],
    target_prompts: List[str],
    source_lang: str,
    target_lang: str,
    num_shots: int,
    num_examples: int = 1000
) -> List[str]:
    """
    Construct few-shot prompts for translation tasks.
    
    Args:
        source_prompts: List of source language sentences
        target_prompts: List of target language sentences (parallel to source_prompts)
        source_lang: Name of source language (e.g., "German")
        target_lang: Name of target language (e.g., "English")
        num_shots: Number of examples to include (0, 1, 5, 10)
        num_examples: Number of examples to process (default: 1000)
        
    Returns:
        List of constructed prompts
    """
    few_shot_prompts = []
    
    for sent in source_prompts[:num_examples]:
        if num_shots > 0:
            # Select num_shots different random examples
            example_indices = random.sample(range(len(source_prompts[:num_examples])), num_shots)
            
            # Build the few-shot examples
            shots = []
            for idx in example_indices:
                example_source = source_prompts[idx]
                example_target = target_prompts[idx]
                shots.append(f"{source_lang}: {example_source}. {target_lang}: {example_target}.")
            
            # Combine the examples
            examples_text = " ".join(shots)
            
            # Construct the prompt
            if num_shots == 1:
                prompt = f"Übersetze. Beispiel: {examples_text} {source_lang}: {sent}. {target_lang}:"
            else:
                prompt = f"Übersetze. Beispiele: {examples_text} Jetzt: {source_lang}: {sent}. {target_lang}:"
        else:
            # Zero-shot case
            prompt = f"Übersetze aus {source_lang} ins {target_lang}: {source_lang}: {sent}. {target_lang}:"
        
        few_shot_prompts.append(prompt)
    
    return few_shot_prompts

def get_topk_and_prob(inputs, tokenizer, model, k=10):
    tokenizer = tokenizer
    model = model 
    
    def get_layer_hidden_states(input_text):
      inputs = tokenizer(input_text, return_tensors="pt")
      inputs = {key: val.to(model.device, dtype=torch.float16) if val.dtype.is_floating_point else val.to(model.device) for key, val in inputs.items()}

      layer_hidden_states = []

      def hook_fn(module, input, output):
        layer_hidden_states.append(output)

      hooks = []
      for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

      with torch.no_grad():
        with autocast(dtype=torch.float16):
            _ = model(**inputs)

      for hook in hooks:
        hook.remove()

      return layer_hidden_states, inputs

    if isinstance(inputs, str):
        inputs = [inputs]

    results = []

    for entry in inputs:
        layer_hidden_states, tokenized_inputs = get_layer_hidden_states(entry)

        for i, hidden_state in enumerate(layer_hidden_states):
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]

            hidden_state_last_token = hidden_state[:, -1, :]
            normalized_hidden = model.model.norm(hidden_state_last_token)
            logits = normalized_hidden @ model.lm_head.weight.T
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            top_k = torch.topk(probs, k, dim=-1)
            top_k_tokens = top_k.indices[0].tolist()
            top_k_probs = top_k.values[0].tolist()
            
            top_k_predictions = [(tokenizer.decode([tok]), prob) for tok, prob in zip(top_k_tokens, top_k_probs)]
            
            results.append({
                "Prompt": entry,
                "Layer": i + 1,
                "Top Predictions": top_k_predictions
            })

    return results

def print_topk(results):
    for result in results:
        print(f"Prompt: {result['Prompt']}")
        print(f"Layer {result['Layer']} Top Predictions:")
        for token, prob in result['Top Predictions']:
            print(f"{token}: {prob:.4f}")
        print()

def save_to_file(file_name, results):
    csv_file = str(file_name)
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Prompt", "Layer", "Top Predictions"])
        writer.writeheader()
        for result in results:
            predictions_str = ", ".join([f"{tok} ({prob:.4f})" for tok, prob in result["Top Predictions"]])
            writer.writerow({
                "Prompt": result["Prompt"],
                "Layer": result["Layer"],
                "Top Predictions": predictions_str
            })

    print(f"Results saved to {csv_file}")

def run_experiments(
    source_prompts: List[str],
    target_prompts: List[str],
    source_lang: str,
    target_lang: str,
    tokenizer,
    model,
    shots: List[int] = [0, 1, 5, 10],
    num_examples: int = 1000,
    top_k: int = 10
) -> None:
    """
    Run translation experiments with different few-shot settings.
    
    Args:
        source_prompts: List of source language sentences
        target_prompts: List of target language sentences
        source_lang: Name of source language
        target_lang: Name of target language
        tokenizer: Model tokenizer
        model: Language model
        shots: List of few-shot settings to run (default: [0, 1, 5, 10])
        num_examples: Number of examples to process (default: 1000)
        top_k: Number of top predictions to save (default: 10)
    """
    for num_shots in shots:
        print(f"Running {num_shots}-shot experiment: {source_lang} -> {target_lang}")
        
        # Construct prompts
        prompts = construct_few_shot_prompts(
            source_prompts,
            target_prompts,
            source_lang,
            target_lang,
            num_shots,
            num_examples
        )
        
        # Get predictions
        results = get_topk_and_prob(prompts, tokenizer, model, k=top_k)
        
        # Save results
        filename = f"{num_shots}shot_{source_lang.lower()}_{target_lang.lower()}_predictions.csv"
        save_to_file(filename, results)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    models_config = {
        "llama-3.1-1b": "meta-llama/Llama-3.1-1B",
        "llama-3.1-3b": "meta-llama/Llama-3.1-3B",
        "llama-3-8b": "meta-llama/Llama-3-8B",
        "aya-101-13b": "CohereForAI/aya-101-13B"
    }
    
    languages = ["englisch", "deutsch", "tschechisch", "arabisch"]
    csv_files = {
        "englisch": "english_flores_prompts.csv",
        "deutsch": "german_flores_prompts(1).csv",
        "tschechisch": "czech_flores_prompts(1).csv",
        "arabisch": "arabic_flores_prompts(1).csv"
    }
    
    # Load prompts
    lang_data = {}
    for lang in languages:
        df = pd.read_csv(csv_files[lang])
        lang_data[lang] = df.iloc[:, 0].tolist()  
    
    # Run experiments for each model
    for model_name, model_path in models_config.items():
        print(f"\n{'='*70}")
        print(f"Loading model: {model_name}")
        print(f"{'='*70}\n")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Run all language combinations
        for source_lang in languages:
            for target_lang in languages:
                if source_lang != target_lang:
                    print(f"\n{'-'*70}")
                    print(f"Model: {model_name} | {source_lang.upper()} -> {target_lang.upper()}")
                    print(f"{'-'*70}")
                    
                    source_prompts = lang_data[source_lang]
                    target_prompts = lang_data[target_lang]
                    
                    run_experiments(
                        source_prompts=source_prompts,
                        target_prompts=target_prompts,
                        source_lang=source_lang.capitalize(),
                        target_lang=target_lang.capitalize(),
                        tokenizer=tokenizer,
                        model=model,
                        shots=[0, 1, 5, 10],
                        num_examples=min(len(source_prompts), 1000),
                        top_k=10
                    )
        
        # Cleanup after each model
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
