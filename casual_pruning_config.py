#!/usr/bin/env python3
"""
Causal Pruning Configuration Framework
=====================================

This module implements a comprehensive framework for causal intervention-based
pruning with integration of Wanda and SparseGPT methods, evaluated across
JMTEB and MTEB benchmark datasets.
"""
# Standard library imports
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import warnings

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from joblib import Parallel, delayed 
from tabulate import tabulate

# Transformers imports
# Note: Explicitly import model classes to prevent lazy loading issues.
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertModel,
    RobertaForSequenceClassification,
    RobertaModel,
)
BATCH_SIZE = 64

warnings.filterwarnings('ignore')

# ============================================================================
# TQDM Logging Handler
# ============================================================================


# Correction: Add a custom logging handler to work seamlessly with tqdm.
class TqdmLoggingHandler(logging.Handler):
    """A logging handler that uses tqdm.write() to prevent progress bar corruption."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


# Configure logging to use the custom Tqdm handler for console output.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Clear existing handlers to prevent duplicate messages
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(logging.FileHandler('comprehensive_validation.log'))
logger.addHandler(TqdmLoggingHandler())
# Prevent log messages from propagating to the root logger
logger.propagate = False


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model specifications."""
    name: str
    model_path: str
    tokenizer_path: str
    max_length: int = 512
    device: str = "auto"
    precision: str = "fp16"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetConfig:
    """Configuration for dataset specifications."""
    name: str
    dataset_path: str
    subset: Optional[str] = None
    task_type: str = "classification"
    metric: str = "accuracy"
    num_samples: Optional[int] = None
    split: str = "test"
    text_columns: List[str] = field(default_factory=lambda: ["text"])
    label_column: str = "label"


@dataclass
class PruningConfig:
    """Configuration for pruning methods."""
    method_name: str
    sparsity_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8]
    )
    causal_masking: bool = False
    structured: bool = False
    global_pruning: bool = True
    importance_metric: str = "causal"


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    models: List[ModelConfig] = field(default_factory=list)
    datasets: List[DatasetConfig] = field(default_factory=list)
    pruning_configs: List[PruningConfig] = field(default_factory=list)
    random_seed: int = 42
    num_runs: int = 3
    statistical_test: str = "wilcoxon"
    significance_level: float = 0.05
    results_dir: str = "/app/results"
    save_intermediate: bool = True


# ============================================================================
# Dataset Configurations
# ============================================================================

def get_jmteb_datasets() -> List[DatasetConfig]:
    """Get all JMTEB dataset configurations."""
    return [
        DatasetConfig(
            name="JMTEB-AmazonCounterfactualClassification",
            dataset_path="sbintuitions/JMTEB",
            subset="amazon_counterfactual_classification",
            task_type="classification", metric="accuracy",
        ),
        DatasetConfig(
            name="JMTEB-AmazonPolarityClassification",
            dataset_path="sbintuitions/JMTEB",
            subset="amazon_polarity_classification",
            task_type="classification", metric="accuracy",
        ),
        DatasetConfig(
            name="JMTEB-AmazonReviewsClassification",
            dataset_path="sbintuitions/JMTEB",
            subset="amazon_review_classification",
            task_type="classification", metric="accuracy",
            text_columns=["review_body"], label_column="stars",
        ),
        DatasetConfig(
            name="JMTEB-MassiveIntentClassification",
            dataset_path="sbintuitions/JMTEB",
            subset="massive_intent_classification",
            task_type="classification", metric="accuracy",
            text_columns=["utt"], label_column="intent",
        ),
        DatasetConfig(
            name="JMTEB-MassiveScenarioClassification",
            dataset_path="sbintuitions/JMTEB",
            subset="massive_scenario_classification",
            task_type="classification", metric="accuracy",
            text_columns=["utt"], label_column="scenario",
        ),
        DatasetConfig(
            name="JMTEB-JSTS", dataset_path="sbintuitions/JMTEB", subset="jsts",
            task_type="similarity", metric="pearson",
            text_columns=["sentence1", "sentence2"], label_column="label",
        ),
        DatasetConfig(
            name="JMTEB-JSICK", dataset_path="sbintuitions/JMTEB", subset="jsick",
            task_type="similarity", metric="pearson",
            text_columns=["sentence1", "sentence2"], label_column="score",
        ),
        DatasetConfig(
            name="JMTEB-LivedoorNewsClustering",
            dataset_path="sbintuitions/JMTEB",
            subset="livedoor_news_clustering",
            task_type="clustering", metric="v_measure", label_column="label",
        ),
        DatasetConfig(
            name="JMTEB-MewsC16JAClustering",
            dataset_path="sbintuitions/JMTEB",
            subset="mewsc16_ja",
            task_type="clustering", metric="v_measure", label_column="label",
        ),
        DatasetConfig(
            name="JMTEB-PawsX",
            dataset_path="sbintuitions/JMTEB", subset="pawsx",
            task_type="pair_classification", metric="accuracy",
            text_columns=["sentence1", "sentence2"], label_column="label",
        ),
        DatasetConfig(
            name="JMTEB-JQaRA",
            dataset_path="sbintuitions/JMTEB", subset="jqara-query", # <-- Corrected
            task_type="retrieval", metric="ndcg_at_10",
            text_columns=["query", "positive", "negative"],
            label_column="label",
        ),
        DatasetConfig(
            name="JMTEB-JaQuAD",
            dataset_path="sbintuitions/JMTEB", subset="jaquad",
            task_type="retrieval", metric="ndcg_at_10",
            text_columns=["question", "context"], label_column="answers",
        ),
        DatasetConfig(
            name="JMTEB-MrTyDi",
            dataset_path="sbintuitions/JMTEB", subset="mrtydi",
            task_type="reranking", metric="map",
            text_columns=["query", "positive", "negative"],
            label_column="label",
        ),
    ]


def get_mteb_datasets() -> List[DatasetConfig]:
    """Get key MTEB dataset configurations."""
    return [
        DatasetConfig(
            name="MTEB-AmazonCounterfactualClassification",
            dataset_path="mteb/amazon_counterfactual",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-AmazonPolarityClassification",
            dataset_path="mteb/amazon_polarity",
            task_type="classification", metric="accuracy",
            text_columns=["content"],
        ),
        DatasetConfig(
            name="MTEB-AmazonReviewsClassification",
            dataset_path="mteb/amazon_reviews_multi",
            task_type="classification", metric="accuracy",
            text_columns=["review_body"], label_column="stars"
        ),
        DatasetConfig(
            name="MTEB-Banking77Classification",
            dataset_path="mteb/banking77",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-EmotionClassification",
            dataset_path="mteb/emotion",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-ImdbClassification",
            dataset_path="mteb/imdb",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-MassiveIntentClassification",
            dataset_path="mteb/mteb/massive_intent",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-MassiveScenarioClassification",
            dataset_path="mteb/mteb/massive_scenario",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-MTOPDomainClassification",
            dataset_path="mteb/mtop_domain",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-MTOPIntentClassification",
            dataset_path="mteb/mtop_intent",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-ToxicConversationsClassification",
            dataset_path="mteb/toxic_conversations_50k",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-TweetSentimentExtractionClassification",
            dataset_path="mteb/tweet_sentiment_extraction",
            task_type="classification", metric="accuracy"
        ),
        DatasetConfig(
            name="MTEB-ArxivClusteringP2P",
            dataset_path="mteb/arxiv-clustering-p2p",
            task_type="clustering", metric="v_measure",
            text_columns=["title", "abstract"]
        ),
        DatasetConfig(
            name="MTEB-BiorxivClusteringP2P",
            dataset_path="mteb/biorxiv-clustering-p2p",
            task_type="clustering", metric="v_measure",
            text_columns=["title", "abstract"]
        ),
        DatasetConfig(
            name="MTEB-TwentyNewsgroupsClustering",
            dataset_path="mteb/twentynewsgroups-clustering",
            task_type="clustering", metric="v_measure",
        ),
        DatasetConfig(
            name="MTEB-SprintDuplicateQuestions",
            dataset_path="mteb/sprintduplicatequestions-pairclassification",
            task_type="pair_classification", metric="accuracy",
            text_columns=["question1", "question2"]
        ),
        DatasetConfig(
            name="MTEB-SciDocsRR",
            dataset_path="mteb/scidocs-reranking",
            task_type="reranking", metric="map",
            text_columns=["query", "positive", "negative"]
        ),
        DatasetConfig(
            name="MTEB-AskUbuntuDupQuestions",
            dataset_path="mteb/askubuntudupquestions-reranking",
            task_type="reranking", metric="map",
            text_columns=["question", "duplicate_question"]
        ),
        DatasetConfig(
            name="MTEB-ArguAna",
            dataset_path="mteb/arguana",
            task_type="retrieval", metric="ndcg_at_10",
            text_columns=["query", "positive", "negative"]
        ),
        DatasetConfig(
            name="MTEB-NFCorpus",
            dataset_path="mteb/nfcorpus",
            task_type="retrieval", metric="ndcg_at_10",
            text_columns=["query", "positive", "negative"]
        ),
        DatasetConfig(
            name="MTEB-STSBenchmark",
            dataset_path="mteb/stsbenchmark-sts",
            task_type="similarity", metric="pearson",
            text_columns=["sentence1", "sentence2"], label_column="score"
        ),
        DatasetConfig(
            name="MTEB-SICKR",
            dataset_path="mteb/sickr-sts",
            task_type="similarity", metric="pearson",
            text_columns=["sentence1", "sentence2"], label_column="score"
        ),
    ]


def get_model_configs() -> List[ModelConfig]:
    """Get model configurations for comprehensive evaluation."""
    return [
        ModelConfig(
            name="Ruri-V2-Base", model_path="cl-nagoya/ruri-base-v2",
            tokenizer_path="cl-nagoya/ruri-base-v2",
        ),
        ModelConfig(
            name="BERT-Base-Japanese-WWM",
            model_path="cl-tohoku/bert-base-japanese-whole-word-masking",
            tokenizer_path="cl-tohoku/bert-base-japanese-whole-word-masking",
        ),
        ModelConfig(
            name="DeBERTa-V2-Base-Japanese",
            model_path="ku-nlp/deberta-v2-base-japanese",
            tokenizer_path="ku-nlp/deberta-v2-base-japanese",
        ),
        ModelConfig(
            name="RoBERTa-Base-Japanese",
            model_path="rinna/japanese-roberta-base",
            tokenizer_path="rinna/japanese-roberta-base",
        ),
        ModelConfig(
            name="BERT-Base-Uncased", model_path="bert-base-uncased",
            tokenizer_path="bert-base-uncased",
        ),
        ModelConfig(
            name="RoBERTa-Base", model_path="roberta-base",
            tokenizer_path="roberta-base",
        ),
        ModelConfig(
            name="DeBERTa-V3-Base",
            model_path="microsoft/deberta-v3-base",
            tokenizer_path="microsoft/deberta-v3-base",
        ),
        ModelConfig(
            name="ELECTRA-Base",
            model_path="google/electra-base-discriminator",
            tokenizer_path="google/electra-base-discriminator",
        ),
        ModelConfig(
            name="mBERT", model_path="bert-base-multilingual-cased",
            tokenizer_path="bert-base-multilingual-cased",
        ),
        ModelConfig(
            name="XLM-RoBERTa-Base", model_path="xlm-roberta-base",
            tokenizer_path="xlm-roberta-base",
        ),
    ]


# ============================================================================
# Causal Importance Computation
# ============================================================================

class CausalImportanceCalculator:
    """
    Computes causal importance using activation patching.
    
    This version implements dynamic scheduling across GPUs to provide a
    detailed progress bar that updates per individual layer.
    """

    def __init__(self, model: nn.Module, tokeniser, device: str = "cuda:0"):
        self.model = model
        self.tokeniser = tokeniser
        self.original_device = device

    def compute_causal_importance(
        self,
        inputs: Dict[str, torch.Tensor],
        target_layers: Optional[List[str]] = None,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute causal importance by parallelising the layer-wise analysis
        across all available GPUs with dynamic scheduling.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                "GPU parallelisation was requested, but no GPUs are available."
            )

        n_gpus = torch.cuda.device_count()
        
        if target_layers is None:
            target_layers = [
                name for name, _ in self.model.named_modules()
                if 'attention' in name or 'layer' in name
            ]
        
        logger.info(
            f"Dynamically dispatching {len(target_layers)} layer analyses "
            f"across {n_gpus} available GPUs..."
        )
        
        # ======================================================================
        # Final Correction: Forcibly synchronise devices before computation.
        # This is a failsafe to ensure the model and inputs are on the same
        # device, regardless of their previous state.
        # ======================================================================
        target_device = inputs.get('input_ids', next(iter(inputs.values()))).device
        self.model.to(target_device)
        # ======================================================================

        # Step 1: Calculate baseline ON THE ORIGINAL DEVICE (GPU).
        with torch.no_grad():
            baseline_outputs = self.model(**inputs)
            if hasattr(baseline_outputs, 'logits'):
                baseline_logits = baseline_outputs.logits
            else:
                baseline_logits = baseline_outputs.last_hidden_state
            # Move the result to the CPU for pickling.
            baseline_probs = torch.softmax(baseline_logits, dim=-1).to('cpu')

        # Step 2: Prepare CPU versions for parallelisation.
        cpu_model = self.model.to('cpu')
        cpu_inputs = {k: v.to('cpu') for k, v in inputs.items()}

        # Create a tqdm progress bar over the total number of layers.
        progress_bar = tqdm(
            target_layers,
            desc="  -> Analysing Layers",
            total=len(target_layers),
            ncols=100
        )

        # Step 3: Define all tasks using only CPU objects.
        tasks = [
            delayed(self._compute_layer_on_gpu)(
                cpu_model, cpu_inputs, layer_name,
                baseline_probs, num_samples, i
            ) for i, layer_name in enumerate(progress_bar)
        ]

        # Step 4: Execute all tasks simultaneously across GPUs.
        results = Parallel(n_jobs=n_gpus, verbose=50)(tasks)

        importance_scores = dict(zip(target_layers, results))
        return importance_scores

    @staticmethod
    def _compute_layer_on_gpu(
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_name: str,
        baseline_probs: torch.Tensor,
        num_samples: Optional[int],
        task_index: int
    ) -> torch.Tensor:
        """
        Compute importance for a single layer on an assigned GPU.
        """
        try:
            # Assign a GPU to the worker process.
            device_id = task_index % torch.cuda.device_count()
            device = f'cuda:{device_id}'

            # The worker receives CPU objects and moves them to its assigned GPU.
            worker_model = model.to(device)
            worker_inputs = {k: v.to(device) for k, v in inputs.items()}
            worker_baseline_probs = baseline_probs.to(device)
        except Exception as e:
            # Fallback to a new logger instance in the child process
            logging.getLogger(__name__).error(f"Error initializing worker: {e}")
            return torch.zeros(1)

        def intervention_hook(module, input, output):
            if isinstance(output, tuple):
                return tuple(
                    (out + torch.randn_like(out) * 0.1) if isinstance(out, torch.Tensor) else out
                    for out in output
                )
            else:
                return output + torch.randn_like(output) * 0.1

        target_layer = None
        for name, module in worker_model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            return torch.zeros(1)

        # If num_samples is None, use the batch size (all items). Otherwise, use num_samples.
        iterations = num_samples if num_samples is not None else inputs['input_ids'].shape[0]
        
        importance_values = []
        for i in range(iterations): # Use the calculated number of iterations
            hook = target_layer.register_forward_hook(intervention_hook)
            try:
                with torch.no_grad():
                    # Process one sample at a time to get per-sample importance
                    single_input = {k: v[i:i+1] for k, v in worker_inputs.items()} 
                    single_baseline_probs = worker_baseline_probs[i:i+1]

                    outputs = worker_model(**single_input)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
                    probs = torch.softmax(logits, dim=-1)
                    
                    kl_div = torch.nn.functional.kl_div(
                        torch.log(probs + 1e-8),
                        single_baseline_probs.to(probs.device), # Ensure devices match
                        reduction='batchmean'
                    )
                    importance_values.append(kl_div.item())
            finally:
                hook.remove()

        return torch.tensor(np.mean(importance_values) if importance_values else 0.0)
    

# ============================================================================
# Advanced Pruning Methods with Causal Masking
# ============================================================================

class WandaWithCausalMasking:
    """Wanda pruning method enhanced with causal masking."""

    def __init__(
        self, model: nn.Module, causal_calculator: CausalImportanceCalculator
    ):
        self.model = model
        self.causal_calculator = causal_calculator

    def compute_wanda_scores(
        self, inputs: Dict[str, torch.Tensor], causal_scores: Dict[str, torch.Tensor], sparsity: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Compute Wanda importance scores with causal masking."""
        
        activations = self._get_activations(inputs)
        wanda_scores = {}

        # --- ▼▼▼ ADD THIS BLOCK FOR CAUSAL MASKING ▼▼▼ ---
        # 1. Determine a threshold to identify causally critical modules.
        # We protect modules with a causal score in the top (1 - sparsity)% percentile.
        if causal_scores:
            all_causal_scores = torch.tensor(list(causal_scores.values()))
            # Protect a slightly larger fraction than the prune ratio for safety
            protection_quantile = sparsity * 0.8 
            protection_threshold = torch.quantile(all_causal_scores, protection_quantile)
        else:
            protection_threshold = float('inf') # No protection if no scores
        # --- ▲▲▲ END OF BLOCK ▲▲▲ ---

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.cpu()
                score = torch.abs(weights) # Default to magnitude if no activations

                if name in activations:
                    act_norms = activations[name]
                    if act_norms.shape[0] == weights.shape[1]:
                        score = torch.abs(weights) * act_norms.unsqueeze(0)
                
                # --- ▼▼▼ ADD THIS BLOCK FOR CAUSAL MASKING ▼▼▼ ---
                # 2. Boost the scores of causally important modules.
                if name in causal_scores and causal_scores[name] > protection_threshold:
                    # Multiply by a large factor to make these scores highly unlikely to be pruned.
                    score *= 10.0 
                # --- ▲▲▲ END OF BLOCK ▲▲▲ ---
                
                wanda_scores[name] = score

        return wanda_scores

    def _get_activations(
        self, inputs: Dict[str, torch.Tensor], batch_size: int = BATCH_SIZE
    ) -> Dict[str, torch.Tensor]:
        """
        Extract INPUT activations from model layers using batch processing.
        This corrected version uses self.model and ensures device consistency.
        """
        activations_cache = {}
        device = self.model.device

        def hook_fn(name):
            def hook(module, input_val, output):
                if isinstance(input_val, tuple) and len(input_val) > 0:
                    if name not in activations_cache:
                        activations_cache[name] = []
                    # Append CPU-bound tensor to save GPU memory
                    activations_cache[name].append(input_val[0].detach().cpu())
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        total_samples = next(iter(inputs.values())).shape[0]
        
        with torch.no_grad():
            # Ensure model is in eval mode for this process
            is_training = self.model.training
            self.model.eval()
            
            for i in range(0, total_samples, batch_size):
                # Move the current batch to the model's device
                batch_inputs = {k: v[i:i+batch_size].to(device) for k, v in inputs.items()}
                self.model(**batch_inputs)

            # Restore model's original training state if necessary
            if is_training:
                self.model.train()

        for hook in hooks:
            hook.remove()
        
        # Aggregate activations from all batches on the CPU
        final_activations = {}
        for name, act_list in activations_cache.items():
            if act_list:
                # Concatenate all batch activations
                concatenated_acts = torch.cat(act_list, dim=0)
                # Calculate the L2 norm for each input neuron, as per the Wanda paper
                # Shape: (batch * sequence_len, in_features) -> sum over dim 0 -> (in_features,)
                if concatenated_acts.dim() == 3: # (batch, seq_len, in_features)
                    reshaped_acts = concatenated_acts.view(-1, concatenated_acts.shape[-1])
                else: # (batch, in_features)
                    reshaped_acts = concatenated_acts
                
                act_norms = torch.linalg.norm(reshaped_acts.float(), ord=2, dim=0)
                final_activations[name] = act_norms

        return final_activations
    
    def prune_with_causal_masking(
        self, inputs: Dict[str, torch.Tensor], sparsity: float
    ) -> nn.Module:
        """Apply Wanda pruning with causal masking."""
        importance_scores = self.compute_wanda_scores(inputs, sparsity)
        # Note: Causal masking logic should be applied here before pruning
        return self._apply_structured_pruning(importance_scores, sparsity)

    def _apply_structured_pruning(
        self, importance_scores: Dict[str, torch.Tensor], sparsity: float
    ) -> nn.Module:
        """Apply structured pruning based on importance scores."""
        for name, module in self.model.named_modules():
            if name in importance_scores and isinstance(module, nn.Linear):
                scores = importance_scores[name].to(module.weight.device)
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask
        return self.model
        

class SparseGPTWithCausalMasking:
    """
    Implements a more faithful, data-driven version of SparseGPT,
    enhanced with causal masking.
    """
    def __init__(
        self, model: nn.Module, causal_calculator: CausalImportanceCalculator
    ):
        self.model = model
        self.causal_calculator = causal_calculator # Used for causal masking
        self.activations = {}

    def _capture_activations(self, inputs: Dict[str, torch.Tensor], batch_size: int = BATCH_SIZE):
        """Captures input activations required to compute the data-driven Hessian."""
        self.activations.clear()
        device = self.model.device

        def hook_fn(name):
            def hook(module, input_val, output):
                if isinstance(input_val, tuple) and len(input_val) > 0:
                    if name not in self.activations:
                        self.activations[name] = []
                    self.activations[name].append(input_val[0].detach().cpu())
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        total_samples = next(iter(inputs.values())).shape[0]
        
        with torch.no_grad():
            is_training = self.model.training
            self.model.eval()
            for i in range(0, total_samples, batch_size):
                batch_inputs = {k: v[i:i+batch_size].to(device) for k, v in inputs.items()}
                self.model(**batch_inputs)
            if is_training:
                self.model.train()

        for hook in hooks:
            hook.remove()

    def compute_sparsegpt_scores(
        self, inputs: Dict[str, torch.Tensor], causal_scores: Dict[str, torch.Tensor], sparsity: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Compute SparseGPT importance scores using a data-driven Hessian and causal masking."""
        logger.info("  -> Applying data-driven SparseGPT-like pruning...")
        self._capture_activations(inputs)
        
        sparsegpt_scores = {}

        if causal_scores:
            all_causal_scores = torch.tensor(list(causal_scores.values()))
            protection_quantile = sparsity * 0.8
            protection_threshold = torch.quantile(all_causal_scores, protection_quantile)
        else:
            protection_threshold = float('inf')

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activations:
                weights = module.weight.data
                act_list = self.activations[name]
                
                if not act_list: continue

                concatenated_acts = torch.cat(act_list, dim=0).to(weights.device)
                
                if concatenated_acts.dim() == 3:
                    reshaped_acts = concatenated_acts.view(-1, concatenated_acts.shape[-1])
                else:
                    reshaped_acts = concatenated_acts

                # H is approximated by 2 * X^T*X. The diagonal of X^T*X is sum(X_i^2).
                H_diag = 2 * torch.sum(reshaped_acts.float().pow(2), dim=0)
                
                # Importance score is W^2 / diag(H_inv) ≈ W^2 * diag(H).
                # Add epsilon for numerical stability.
                score = weights.pow(2) / (H_diag.unsqueeze(0) + 1e-8)
                
                if name in causal_scores and causal_scores[name] > protection_threshold:
                    score *= 10.0 # Boost score for causal protection
                
                sparsegpt_scores[name] = score
        
        return sparsegpt_scores

    def prune_with_causal_masking(
        self, inputs: Dict[str, torch.Tensor], causal_scores: Dict[str, torch.Tensor], sparsity: float
    ) -> nn.Module:
        """Apply SparseGPT pruning with causal masking."""
        importance_scores = self.compute_sparsegpt_scores(inputs, causal_scores, sparsity)
        
        # In a full implementation, a complex weight update would happen here.
        # For this framework, we use simple masking based on the improved scores.
        for name, module in self.model.named_modules():
            if name in importance_scores and isinstance(module, nn.Linear):
                scores = importance_scores[name].to(module.weight.device)
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask
        
        return self.model

class WandaPruner:
    """
    Implements the Wanda pruning method rigorously, following the original paper.
    Key correction: Uses L2-norm of activations instead of mean of absolute values.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}

    def _capture_activations(self, inputs: Dict[str, torch.Tensor], batch_size: int = BATCH_SIZE):
        """
        Runs a forward pass to capture input activations for each linear layer,
        processing in batches to conserve memory.
        """
        self.activations.clear()
        
        def hook_fn(name):
            def hook(module, input_val, output):
                if isinstance(input_val, tuple) and len(input_val) > 0:
                    if name not in self.activations:
                        self.activations[name] = []
                    # Store on CPU to avoid accumulating tensors on GPU
                    self.activations[name].append(input_val[0].detach().cpu())
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        total_samples = next(iter(inputs.values())).shape[0]
        
        with torch.no_grad():
            original_training_state = self.model.training
            self.model.eval()
            
            for i in range(0, total_samples, batch_size):
                batch_inputs = {k: v[i:i+batch_size].to(self.model.device) for k, v in inputs.items()}
                self.model(**batch_inputs)
            
            if original_training_state:
                self.model.train()

        for hook in hooks:
            hook.remove()

    def prune(self, inputs: Dict[str, torch.Tensor], sparsity: float) -> nn.Module:
        """Applies Wanda pruning to the model."""
        logger.info("  -> Applying rigorous Wanda pruning...")
        self._capture_activations(inputs)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activations:
                weights = module.weight.data
                act_list = self.activations[name]
                
                if not act_list:
                    continue
                
                # Concatenate all captured activation batches on CPU
                concatenated_acts = torch.cat(act_list, dim=0)
                
                # Reshape activations to (num_tokens, in_features)
                if concatenated_acts.dim() == 3:
                    reshaped_acts = concatenated_acts.view(-1, concatenated_acts.shape[-1])
                else:
                    reshaped_acts = concatenated_acts
                
                # Calculate the L2 norm for each input neuron across all tokens.
                act_norms = torch.linalg.norm(reshaped_acts.float(), ord=2, dim=0).to(weights.device)
                
                # Calculate importance scores: |W| * ||X||
                scores = torch.abs(weights) * act_norms.unsqueeze(0) # unsqueeze for broadcasting
                
                # Determine threshold and create mask
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                
                # Apply the mask to prune weights
                module.weight.data *= mask
                
        return self.model

class SparseGPTPruner:
    """
    Implements a more faithful, data-driven version of SparseGPT.
    Key correction: Computes an approximate Hessian diagonal from actual
    calibration data (X^T * X), rather than using a dummy value.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}

    def _capture_activations(self, inputs: Dict[str, torch.Tensor], batch_size: int = BATCH_SIZE):
        """
        Captures input activations, similar to the Wanda pruner.
        This is necessary to compute the data-driven Hessian.
        """
        self.activations.clear()
        
        def hook_fn(name):
            def hook(module, input_val, output):
                if isinstance(input_val, tuple) and len(input_val) > 0:
                    if name not in self.activations:
                        self.activations[name] = []
                    self.activations[name].append(input_val[0].detach().cpu())
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        total_samples = next(iter(inputs.values())).shape[0]
        
        with torch.no_grad():
            original_training_state = self.model.training
            self.model.eval()

            for i in range(0, total_samples, batch_size):
                batch_inputs = {k: v[i:i+batch_size].to(self.model.device) for k, v in inputs.items()}
                self.model(**batch_inputs)

            if original_training_state:
                self.model.train()

        for hook in hooks:
            hook.remove()

    def prune(self, inputs: Dict[str, torch.Tensor], sparsity: float) -> nn.Module:
        """Applies a more rigorous SparseGPT-like pruning to the model."""
        logger.info("  -> Applying data-driven SparseGPT-like pruning...")
        # SparseGPT requires calibration data to compute the Hessian.
        self._capture_activations(inputs)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activations:
                weights = module.weight.data
                act_list = self.activations[name]
                
                if not act_list:
                    continue

                concatenated_acts = torch.cat(act_list, dim=0).to(weights.device)
                
                if concatenated_acts.dim() == 3:
                    reshaped_acts = concatenated_acts.view(-1, concatenated_acts.shape[-1])
                else:
                    reshaped_acts = concatenated_acts

                # The Hessian H is approximated by 2 * X^T*X. We need its inverse diagonal.
                # The diagonal of X^T*X is sum(X_i^2) for each feature i.
                H_diag = 2 * torch.sum(reshaped_acts.float().pow(2), dim=0)
                
                # The importance score is W^2 / diag(H_inv), which is approx. W^2 * diag(H).
                scores = weights.pow(2) / (H_diag.unsqueeze(0) + 1e-8) # Add epsilon for numerical stability    
                
                # Mask and prune
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask

                # Note: A full SparseGPT implementation includes a complex weight update step here
                # using the off-diagonal Hessian information. This is omitted for simplicity
                # but the core importance scoring is now correctly data-driven.

        return self.model
    

# ============================================================================
# Evaluation Framework
# ============================================================================

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for JMTEB and MTEB datasets."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def evaluate_model_on_dataset(
        self,
        model: nn.Module,
        tokeniser,
        dataset_config: DatasetConfig,
        pruning_config: PruningConfig
    ) -> Dict[str, float]:
        """Evaluate a single model on a single dataset."""
        try:
            dataset = self._load_dataset(dataset_config)
            task_map = {
                "classification": self._evaluate_classification,
                "similarity": self._evaluate_similarity,
                "clustering": self._evaluate_clustering,
                "retrieval": self._evaluate_retrieval,
                "reranking": self._evaluate_reranking,
                "pair_classification": self._evaluate_pair_classification,
            }
            eval_func = task_map.get(dataset_config.task_type)

            if eval_func:
                return eval_func(model, tokeniser, dataset, dataset_config)

            logger.warning(f"Unknown task type: {dataset_config.task_type}")
            return {"score": 0.0}

        except Exception as e:
            logger.error(f"Error evaluating {dataset_config.name}: {e}")
            return {"score": 0.0, "error": str(e)}

    def _load_dataset(self, config: DatasetConfig):
        """Load dataset from Hugging Face."""
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                config.dataset_path,
                config.subset,
                split=config.split,
                trust_remote_code=True
            )
            if config.num_samples and len(dataset) > config.num_samples:
                dataset = dataset.shuffle(seed=42).select(
                    range(config.num_samples)
                )
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset {config.name}: {e}")
            return None

    def _evaluate_classification(
        self, model: nn.Module, tokeniser, dataset, config: DatasetConfig
    ) -> Dict[str, float]:
        """Evaluate classification task."""
        if dataset is None:
            return {"accuracy": 0.0, "f1": 0.0, "samples": 0}
        predictions, true_labels = [], []

        eval_desc = "  -> Evaluating"
        for example in tqdm(dataset, desc=eval_desc, leave=False, ncols=80):
            try:
                text = (
                    example[config.text_columns[0]]
                    if len(config.text_columns) == 1 else
                    " ".join([str(example[c]) for c in config.text_columns])
                )
                inputs = tokeniser(
                    text, max_length=512, truncation=True, padding=True,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=-1).item()
                predictions.append(pred)
                true_labels.append(example[config.label_column])
            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                continue
        if predictions:
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')
            return {"accuracy": accuracy, "f1": f1, "samples": len(predictions)}
        return {"accuracy": 0.0, "f1": 0.0, "samples": 0}

    def _evaluate_similarity(
        self, model: nn.Module, tokeniser, dataset, config: DatasetConfig
    ) -> Dict[str, float]:
        """Evaluate similarity task."""
        if dataset is None:
            return {"pearson": 0.0, "samples": 0}
        predictions, true_scores = [], []
        for example in tqdm(dataset, desc="  -> Evaluating", leave=False, ncols=80):
            try:
                sent1 = example[config.text_columns[0]]
                sent2 = example[config.text_columns[1]]
                emb1 = self._get_embedding(model, tokeniser, sent1)
                emb2 = self._get_embedding(model, tokeniser, sent2)
                similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
                predictions.append(similarity)
                true_scores.append(float(example[config.label_column]))
            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                continue
        if len(predictions) > 1:
            pearson_corr, _ = stats.pearsonr(true_scores, predictions)
            return {"pearson": pearson_corr, "samples": len(predictions)}
        return {"pearson": 0.0, "samples": len(predictions)}

    def _get_embedding(
        self, model: nn.Module, tokeniser, text: str
    ) -> torch.Tensor:
        """Get text embedding from model."""
        inputs = tokeniser(
            text, max_length=512, truncation=True, padding=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                embedding = embeddings.mean(dim=1).squeeze()
            else:
                embedding = outputs.pooler_output.squeeze()
        return embedding

    def _evaluate_clustering(
        self, model: nn.Module, tokeniser, dataset, config: DatasetConfig
    ) -> Dict[str, float]:
        """Evaluate clustering task."""
        if dataset is None:
            return {"v_measure": 0.0, "samples": 0}
        embeddings, true_labels = [], []
        for example in tqdm(dataset, desc="  -> Evaluating", leave=False, ncols=80):
            try:
                text = (
                    example[config.text_columns[0]]
                    if len(config.text_columns) == 1 else
                    " ".join([str(example[c]) for c in config.text_columns])
                )
                embedding = self._get_embedding(model, tokeniser, text)
                embeddings.append(embedding.cpu().numpy())
                true_labels.append(example[config.label_column])
            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                continue
        if embeddings:
            from sklearn.cluster import KMeans
            from sklearn.metrics import v_measure_score
            embeddings_arr = np.array(embeddings)
            n_clusters = len(set(true_labels))
            if n_clusters < 2:
                return {"v_measure": 0.0, "samples": len(embeddings)}
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            pred_labels = kmeans.fit_predict(embeddings_arr)
            v_measure = v_measure_score(true_labels, pred_labels)
            return {"v_measure": v_measure, "samples": len(embeddings)}
        return {"v_measure": 0.0, "samples": 0}

    def _evaluate_retrieval(
        self, model: nn.Module, tokeniser, dataset, config: DatasetConfig
    ) -> Dict[str, float]:
        """Placeholder for retrieval task evaluation."""
        logger.warning("Retrieval evaluation is a simplified placeholder.")
        return {"ndcg_at_10": 0.5, "samples": len(dataset) if dataset else 0}

    def _evaluate_reranking(
        self, model: nn.Module, tokeniser, dataset, config: DatasetConfig
    ) -> Dict[str, float]:
        """Placeholder for reranking task evaluation."""
        logger.warning("Reranking evaluation is a simplified placeholder.")
        return {"map": 0.5, "samples": len(dataset) if dataset else 0}

    def _evaluate_pair_classification(
        self, model: nn.Module, tokeniser, dataset, config: DatasetConfig
    ) -> Dict[str, float]:
        """Evaluate pair classification task."""
        return self._evaluate_classification(model, tokeniser, dataset, config)


# ============================================================================
# Statistical Analysis Framework
# ============================================================================

class StatisticalAnalyser:
    """Performs statistical analysis for experimental results."""

    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df

    def perform_significance_tests(
        self,
        method_col: str = "pruning_method",
        perf_col: str = "performance",
        baseline: str = "Magnitude"
    ) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance tests against a baseline."""
        results = {}
        methods = self.results_df[method_col].unique()
        baseline_scores = self.results_df[
            self.results_df[method_col] == baseline
        ][perf_col].values

        for method in methods:
            if method == baseline:
                continue
            method_scores = self.results_df[
                self.results_df[method_col] == method
            ][perf_col].values

            if len(method_scores) > 1 and len(baseline_scores) > 1:
                try:
                    stat, p_val = stats.wilcoxon(
                        method_scores, baseline_scores, zero_method='pratt'
                    )
                    effect = self._compute_effect_size(method_scores, baseline_scores)
                    results[method] = {
                        "p_value": p_val, "statistic": stat,
                        "effect_size": effect, "significant": p_val < 0.05,
                        "mean_difference": np.mean(method_scores) - np.mean(baseline_scores)
                    }
                except Exception as e:
                    logger.warning(f"Could not perform test for {method}: {e}")
                    results[method] = {
                        "p_value": 1.0, "statistic": 0.0, "effect_size": 0.0,
                        "significant": False, "mean_difference": 0.0
                    }
        return results

    def _compute_effect_size(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> float:
        """Compute Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Guard against division by zero if variances are zero
        if (n1 + n2 - 2) == 0:
            return 0.0
            
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive statistical summary."""
        summary = {"descriptive_statistics": {}}
        for method in self.results_df["pruning_method"].unique():
            method_data = self.results_df[
                self.results_df["pruning_method"] == method
            ]["performance"]
            if not method_data.empty and len(method_data) > 0:
                mean, std, count = (
                    method_data.mean(), method_data.std(), len(method_data)
                )
                if count > 0:
                    ci_factor = 1.96 * std / np.sqrt(count)
                    summary["descriptive_statistics"][method] = {
                        "mean": mean, "std": std, "median": method_data.median(),
                        "min": method_data.min(), "max": method_data.max(),
                        "count": count, "ci_95_lower": mean - ci_factor,
                        "ci_95_upper": mean + ci_factor
                    }
        summary["significance_tests"] = self.perform_significance_tests()
        return summary


# ============================================================================
# Comprehensive Execution Framework
# ============================================================================

class ComprehensiveValidationFramework:
    """Main framework for comprehensive validation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        self.evaluator = ComprehensiveEvaluator(config)
        self.importance_cache = {}  # Correction: Add a cache for importance scores.
        self.cache_dir = Path(self.config.results_dir) / "importance_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run the complete validation with an efficient caching mechanism for
        importance scores.
        """
        logger.info("Starting comprehensive validation...")

        all_experiments = []
        for model_cfg in self.config.models:
            for data_cfg in self.config.datasets:
                for prune_cfg in self.config.pruning_configs:
                    for sparsity in prune_cfg.sparsity_levels:
                        for run in range(self.config.num_runs):
                            all_experiments.append(
                                (model_cfg, data_cfg, prune_cfg, sparsity, run)
                            )
        
        logger.info(f"Total experiments: {len(all_experiments)}")
        
        seen_pairs = set()
        unique_model_dataset_pairs = []
        for model_cfg, data_cfg, _, _, _ in all_experiments:
            pair_identifier = (model_cfg.name, data_cfg.name)
            if pair_identifier not in seen_pairs:
                unique_model_dataset_pairs.append((model_cfg, data_cfg))
                seen_pairs.add(pair_identifier)

        for model_config, dataset_config in unique_model_dataset_pairs:
            cache_key = (model_config.name, dataset_config.name)
            sane_model_name = model_config.name.replace("/", "_")
            sane_dataset_name = dataset_config.name.replace("/", "_")
            cache_file = self.cache_dir / f"{sane_model_name}_{sane_dataset_name}.pt"

            if cache_file.exists():
                logger.info(f"Loading cached scores from: {cache_file}")
                self.importance_cache[cache_key] = torch.load(cache_file)
                continue

            logger.info(
                f"\nNo cache found. Calculating scores for {model_config.name} "
                f"on {dataset_config.name}..."
            )
            
            temp_model, temp_tokeniser = self._load_model_and_tokeniser(
                model_config, dataset_config.task_type
            )
            calib_dataset = self.evaluator._load_dataset(dataset_config)
            calib_inputs = self._prepare_calibration_data(
                calib_dataset, temp_tokeniser, dataset_config.text_columns
            )
            device = next(temp_model.parameters()).device
            inputs = {k: v.to(device) for k, v in calib_inputs.items()}

            causal_calc = CausalImportanceCalculator(
                temp_model, temp_tokeniser, device=device
            )
            causal_scores = causal_calc.compute_causal_importance(inputs)
            
            wanda_pruner = WandaWithCausalMasking(temp_model, causal_calc)
            wanda_scores = {}
            activations = wanda_pruner._get_activations(inputs)
            for name, module in temp_model.named_modules():
                if isinstance(module, nn.Linear):
                    weights = module.weight.data
                    if name in activations:
                        acts = activations[name]
                        
                        # Check tensor dimensions before calculating the mean
                        if acts.dim() == 3:
                            act_norm = torch.abs(acts).mean(dim=[0, 1])
                        elif acts.dim() == 2:
                            act_norm = torch.abs(acts).mean(dim=0)
                        else:
                            # Fallback for unexpected dimensions
                            wanda_scores[name] = torch.abs(weights)
                            continue

                        # Now act_norm is guaranteed to be a 1D tensor, so the shape check is safe
                        if act_norm.shape[0] == weights.shape[1]:
                            wanda_scores[name] = torch.abs(weights) * act_norm
                        else:
                            wanda_scores[name] = torch.abs(weights)
                    else:
                        wanda_scores[name] = torch.abs(weights)

            scores_to_cache = {"causal": causal_scores, "wanda": wanda_scores}
            self.importance_cache[cache_key] = scores_to_cache
            logger.info(f"Saving new scores to: {cache_file}")
            torch.save(scores_to_cache, cache_file)
            del temp_model
        
        progress_bar = tqdm(all_experiments, desc="Overall Progress", ncols=80)
        for exp_params in progress_bar:
            model_cfg, data_cfg, prune_cfg, sparsity, run = exp_params
            try:
                progress_bar.set_description(
                    f"Running {model_cfg.name} on {data_cfg.name}"
                )
                model, tokeniser = self._load_model_and_tokeniser(
                    model_cfg, data_cfg.task_type
                )
                
                if sparsity > 0:
                    cache_key = (model_cfg.name, data_cfg.name)
                    cached_scores = self.importance_cache[cache_key]
                    
                    # Regenerate calibration inputs for methods that need them on-the-fly
                    calib_dataset = self.evaluator._load_dataset(data_cfg)
                    calibration_inputs = self._prepare_calibration_data(
                        calib_dataset, tokeniser, data_cfg.text_columns
                    )
                    pruned_model = self._apply_pruning(
                        model,
                        tokeniser,
                        prune_cfg,
                        sparsity,
                        cached_scores["causal"],
                        cached_scores["wanda"],
                        calibration_inputs  # <-- Pass the inputs as a new argument
                    )
                else:
                    pruned_model = model

                eval_results = self.evaluator.evaluate_model_on_dataset(
                    pruned_model, tokeniser, data_cfg, prune_cfg
                )
                
                result = {
                    "model": model_cfg.name, "dataset": data_cfg.name,
                    "task_type": data_cfg.task_type,
                    "pruning_method": prune_cfg.method_name,
                    "sparsity": sparsity, "run": run,
                    "performance": eval_results.get(data_cfg.metric, 0.0),
                    "additional_metrics": eval_results,
                    "timestamp": pd.Timestamp.now(),
                }
                self.results.append(result)

                if (self.config.save_intermediate and 
                        progress_bar.n > 0 and progress_bar.n % 10 == 0):
                    self._save_intermediate_results()

            except Exception as e:
                logger.error(
                    f"Experiment failed for {model_cfg.name} on {data_cfg.name}: {e}",
                    exc_info=True
                )
                continue
        
        final_results = self._compile_results()
        self._save_final_results(final_results)
        logger.info("Comprehensive validation completed!")
        return final_results
                

    def _report_model_size(self, model: nn.Module, description: str):
        """Calculates and logs the model's size and parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum(p.nonzero().size(0) for p in model.parameters())
        
        # Calculate size in MB by saving to a temporary buffer
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.getbuffer().nbytes / 1e6

        sparsity = (1 - (nonzero_params / total_params)) * 100

        logger.info(f"    -> {description}:")
        logger.info(f"       Model Size: {size_mb:.2f} MB")
        logger.info(f"       Total Params: {total_params:,}")
        logger.info(f"       Non-Zero Params: {nonzero_params:,} ({100-sparsity:.1f}%)")
        logger.info(f"       Sparsity: {sparsity:.2f}%")


    def _run_single_experiment(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        pruning_config: PruningConfig,
        sparsity: float,
        run: int,
    ) -> Dict[str, Any]:
        """Run a single experiment."""
        run_desc = (
            f"{model_config.name} | {dataset_config.name} | "
            f"{pruning_config.method_name} | {sparsity} | Run {run}"
        )
        logger.info(f"Running: {run_desc}")
        try:
            model, tokeniser = self._load_model_and_tokeniser(
                model_config, dataset_config.task_type
            )
            if sparsity > 0:
                self._report_model_size(model, "Original Model")

            if sparsity > 0:
                calib_dataset = self.evaluator._load_dataset(dataset_config)
                calib_inputs = self._prepare_calibration_data(
                    calib_dataset, tokeniser, dataset_config.text_columns
                )
                model = self._apply_pruning(
                    model, tokeniser, pruning_config, sparsity, calib_inputs
                )
                self._report_model_size(model, f"Pruned Model ({pruning_config.method_name} @ {sparsity*100:.0f}%)")
            
            eval_results = self.evaluator.evaluate_model_on_dataset(
                model, tokeniser, dataset_config, pruning_config
            )
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)
            actual_sparsity = 1.0 - (nonzero_params / total_params)

            result = {
                "model": model_config.name,
                "dataset": dataset_config.name,
                "task_type": dataset_config.task_type,
                "pruning_method": pruning_config.method_name,
                "sparsity": sparsity,
                "run": run,
                "target_sparsity": sparsity, 
                "actual_sparsity": actual_sparsity, 
                "performance": eval_results.get(dataset_config.metric, 0.0),
                "additional_metrics": eval_results,
                "timestamp": pd.Timestamp.now(),
            }
            logger.info(
                f"  -> Completed. Score ({dataset_config.metric}): "
                f"{result['performance']:.4f}"
            )
            return result
        except Exception as e:
            logger.error(f"Single experiment failed: {e}", exc_info=True)
            return {
                "model": model_config.name, "dataset": dataset_config.name,
                "task_type": dataset_config.task_type,
                "pruning_method": pruning_config.method_name,
                "sparsity": sparsity, "run": run, "performance": 0.0,
                "error": str(e), "timestamp": pd.Timestamp.now()
            }

    def _prepare_calibration_data(
        self, dataset, tokeniser, text_columns: List[str], num_samples: int = None
    ) -> Dict[str, torch.Tensor]:
        """Create a single batch of calibration data from a dataset."""
        if not dataset or len(dataset) == 0:
            sample_text = ["This is a sample text for importance computation."]
            return tokeniser(
                sample_text, max_length=512, truncation=True,
                padding=True, return_tensors="pt"
            )
        
        if num_samples is None:
            effective_num_samples = len(dataset)
        else:
            effective_num_samples = min(num_samples, len(dataset))
        calibration_dataset = dataset.select(range(effective_num_samples))
        
        logger.info(f"  -> Using {len(calibration_dataset)} samples for calibration.")
        
        text_list = []
        calib_desc = "  -> Preparing calibration data"
        for example in tqdm(calibration_dataset, desc=calib_desc, leave=False, ncols=100):
            try:
                # --- THIS IS THE CORRECTED LOGIC ---
                texts_to_join = []
                for col in text_columns:
                    if col in example and example[col] is not None:
                        # Handle simple, flat columns
                        texts_to_join.append(str(example[col]))
                    elif col == 'positive' and 'passages' in example and 'positive_ctxs' in example['passages']:
                        # Special handling for JQaRA 'positive' column
                        # Take the first positive context if it exists
                        if example['passages']['positive_ctxs']:
                            texts_to_join.append(str(example['passages']['positive_ctxs'][0].get('text', '')))
                    elif col == 'negative' and 'passages' in example and 'negative_ctxs' in example['passages']:
                        # Special handling for JQaRA 'negative' column
                        # Take the first negative context if it exists
                        if example['passages']['negative_ctxs']:
                           texts_to_join.append(str(example['passages']['negative_ctxs'][0].get('text', '')))
                
                text_list.append(" ".join(texts_to_join))
                # --- END OF CORRECTED LOGIC ---

            except KeyError as e:
                logger.warning(f"Skipping example due to missing key: {e}")
                continue
        
        if not text_list:
            raise ValueError("Failed to prepare any calibration data. Check dataset structure and text_columns.")

        return tokeniser(
            text_list, max_length=512, truncation=True,
            padding="max_length", return_tensors="pt"
        )

    def _load_model_and_tokeniser(
        self, model_config: ModelConfig, task_type: str
    ):
        """Load a model and tokeniser based on the task type."""
        try:
            tokeniser = AutoTokenizer.from_pretrained(model_config.tokenizer_path)
            if task_type in ["classification", "pair_classification"]:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_config.model_path
                )
            else:
                model = AutoModel.from_pretrained(model_config.model_path)
            model.to(model_config.device)
            model.eval()
            return model, tokeniser
        except Exception as e:
            logger.error(f"Failed to load model {model_config.name}: {e}")
            raise

    def _apply_hierarchical_pruning(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        causal_scores: Dict[str, torch.Tensor],
        sparsity: float,
        base_method: str = "Wanda"
    ) -> nn.Module:
        """
        Applies pruning only within causally important modules (Hypothesis 3).
        """
        logger.info(f"  -> Applying Hierarchical Causal Pruning with base method: {base_method}...")
        
        # 1. Identify causally important modules to serve as the pruning scope.
        protected_modules = set()
        if causal_scores:
            all_causal_scores = torch.tensor(list(causal_scores.values()))
            # We want to keep (1 - sparsity) modules, so we find the threshold at the sparsity level.
            # Modules with a score *above* this threshold will be the ones we prune *inside*.
            protection_threshold = torch.quantile(all_causal_scores, sparsity)
            
            for name, score in causal_scores.items():
                if score > protection_threshold:
                    protected_modules.add(name)
        
        if not protected_modules:
            logger.warning("No modules identified for hierarchical pruning. Returning original model.")
            return model
            
        logger.info(f"     Pruning will be applied inside these {len(protected_modules)} modules.")

        # 2. Instantiate a pruner for the base method (Wanda or SparseGPT)
        if base_method == "Wanda":
            pruner = WandaPruner(model)
            pruner._capture_activations(inputs)
        elif base_method == "SparseGPT":
            pruner = SparseGPTPruner(model)
            pruner._capture_activations(inputs)
        else:
            raise ValueError(f"Unknown base method for hierarchical pruning: {base_method}")

        # 3. Iterate through modules and apply pruning ONLY if they are in the protected set.
        for name, module in pruner.model.named_modules():
            if isinstance(module, nn.Linear) and name in pruner.activations and name in protected_modules:
                weights = module.weight.data
                act_list = pruner.activations[name]
                if not act_list: continue
                
                concatenated_acts = torch.cat(act_list, dim=0)
                if concatenated_acts.dim() == 3:
                    reshaped_acts = concatenated_acts.view(-1, concatenated_acts.shape[-1])
                else:
                    reshaped_acts = concatenated_acts
                
                # Calculate scores based on the chosen method
                if base_method == "Wanda":
                    act_norms = torch.linalg.norm(reshaped_acts.float(), ord=2, dim=0).to(weights.device)
                    scores = torch.abs(weights) * act_norms.unsqueeze(0)
                elif base_method == "SparseGPT":
                    H_diag = 2 * torch.sum(reshaped_acts.float().pow(2), dim=0).to(weights.device)
                    scores = weights.pow(2) / (H_diag.unsqueeze(0) + 1e-8)

                # Prune weights *within* this important module
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask

        return pruner.model

    def _apply_pruning(
        self,
        model: nn.Module,
        tokeniser,
        pruning_config: PruningConfig,
        sparsity: float,
        causal_scores: Dict[str, torch.Tensor],
        wanda_scores: Dict[str, torch.Tensor],
        calibration_inputs: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Apply the specified pruning method using pre-computed scores or live inputs."""
        device = next(model.parameters()).device
        method = pruning_config.method_name
        pruned_model = copy.deepcopy(model)
        inputs = {k: v.to(device) for k, v in calibration_inputs.items()}

        if method == "Causal":
            logger.info("  -> Applying pure Causal pruning...")
            if not causal_scores:
                logger.warning("Causal scores not found. Returning original model.")
                return model
            
            all_scores = torch.tensor(list(causal_scores.values()))
            threshold = torch.quantile(all_scores, sparsity)

            for name, module in pruned_model.named_modules():
                if name in causal_scores:
                    if causal_scores[name] < threshold:
                        if hasattr(module, 'weight') and module.weight is not None:
                            module.weight.data.zero_()
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data.zero_()
            return pruned_model
            
        elif method == "Magnitude":
            logger.info("  -> Applying Magnitude pruning...")
            for _, module in pruned_model.named_modules():
                if isinstance(module, nn.Linear):
                    scores = torch.abs(module.weight.data)
                    threshold = torch.quantile(scores.flatten(), sparsity)
                    module.weight.data *= (scores > threshold).float()
            return pruned_model

        elif method == "Gradient":
            logger.info("  -> Applying Gradient pruning...")
            pruned_model.train()

            batch_size = BATCH_SIZE
            total_samples = inputs['input_ids'].shape[0]
            pruned_model.zero_grad()

            for i in tqdm(range(0, total_samples, batch_size), desc="  -> Calculating Gradients", leave=False, ncols=100):
                batch_inputs = {k: v[i:i + batch_size] for k, v in inputs.items()}
                outputs = pruned_model(**batch_inputs)
                
                if hasattr(outputs, 'logits'):
                    loss = outputs.logits.sum()
                else:
                    loss = outputs.last_hidden_state.sum()
                
                loss.backward()
            
            for _, module in pruned_model.named_modules():
                if isinstance(module, nn.Linear) and hasattr(module.weight, 'grad') and module.weight.grad is not None:
                    scores = torch.abs(module.weight.grad * module.weight.data) # SNIP-like score
                    threshold = torch.quantile(scores.flatten(), sparsity)
                    module.weight.data *= (scores > threshold).float()

            pruned_model.zero_grad(set_to_none=True)
            pruned_model.eval()
            return pruned_model
        
        elif method == "Wanda":
            pruner = WandaPruner(pruned_model)
            return pruner.prune(inputs, sparsity)
        
        elif method == "SparseGPT":
            pruner = SparseGPTPruner(pruned_model)
            return pruner.prune(inputs, sparsity)

        elif method == "CausalMaskedWanda":
            wanda_pruner = WandaWithCausalMasking(pruned_model, None)
            scores = wanda_pruner.compute_wanda_scores(inputs, causal_scores, sparsity)
            return wanda_pruner._apply_structured_pruning(scores, sparsity)

        # --- ▼▼▼ THIS IS THE CORRECTED BLOCK ▼▼▼ ---
        elif method == "CausalMaskedSparseGPT":
            # The causal_calculator is not directly needed if we pass scores, so pass None.
            sparse_pruner = SparseGPTWithCausalMasking(pruned_model, None)
            
            # The prune_with_causal_masking method now correctly receives all its arguments.
            return sparse_pruner.prune_with_causal_masking(inputs, causal_scores, sparsity)
        # --- ▲▲▲ END OF CORRECTION ▲▲▲ ---

        elif method == "HierarchicalCausalWanda":
            return self._apply_hierarchical_pruning(
                pruned_model, inputs, causal_scores, sparsity, base_method="Wanda"
            )

        elif method == "HierarchicalCausalSparseGPT":
            return self._apply_hierarchical_pruning(
                pruned_model, inputs, causal_scores, sparsity, base_method="SparseGPT"
            )
        
        logger.warning(
            f"Pruning method '{method}' not found. Returning original model."
        )
        return model
    
    def _apply_wanda_from_scores(
        self, model: nn.Module, sparsity: float, wanda_scores: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Apply Wanda pruning using pre-computed scores."""
        logger.info("   -> Applying Wanda pruning from cached scores...")
        for name, module in model.named_modules():
            if name in wanda_scores and isinstance(module, nn.Linear):
                scores = wanda_scores[name].to(module.weight.device)
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask
        return model

    def _apply_causal_wanda_from_scores(
        self, model: nn.Module, sparsity: float, 
        causal_scores: Dict[str, torch.Tensor], wanda_scores: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Apply Causal Wanda pruning using pre-computed scores."""
        logger.info("   -> Applying CausalMaskedWanda from cached scores...")
        for name, module in model.named_modules():
            if name in wanda_scores and isinstance(module, nn.Linear):
                scores = wanda_scores[name].to(module.weight.device)
                
                # Apply causal masking to boost important weights
                if name in causal_scores:
                    causal_importance = causal_scores[name].to(scores.device)
                    # This logic assumes causal_importance is a scalar per layer; adjust if it's per-weight
                    if causal_importance.numel() == 1:
                        protection_threshold = torch.quantile(
                            torch.tensor(list(causal_scores.values())), 1 - sparsity * 0.5
                        )
                        if causal_importance > protection_threshold:
                            scores *= 1.5 # Boost scores for causally important layers
                
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask
        return model
                
    def _get_activations(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Helper to extract activations from model layers."""
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    activations[name] = output[0].detach()
            return hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        with torch.no_grad():
            model(**inputs)
        for hook in hooks:
            hook.remove()
        return activations
    
    def _apply_causal_pruning(
        self, model, tokeniser, inputs, sparsity, device
    ):
        """Apply pruning based on causal importance scores."""
        causal_calc = CausalImportanceCalculator(model, tokeniser, device=device)
        importance_scores = causal_calc.compute_causal_importance(inputs)
        for name, module in model.named_modules():
            if name in importance_scores and isinstance(module, nn.Linear):
                scores = importance_scores[name]
                threshold = torch.quantile(scores.flatten(), sparsity)
                module.weight.data *= (scores > threshold).float()
        return model

    def _apply_causal_wanda(
        self, model, tokeniser, inputs, sparsity, device
    ):
        """Apply Wanda pruning with causal masking."""
        causal_calc = CausalImportanceCalculator(model, tokeniser, device=device)
        wanda_pruner = WandaWithCausalMasking(model, causal_calc)
        return wanda_pruner.prune_with_causal_masking(inputs, sparsity)

    def _apply_causal_sparsegpt(
        self, model, tokeniser, inputs, sparsity, device
    ):
        """Apply SparseGPT pruning with causal masking."""
        causal_calc = CausalImportanceCalculator(model, tokeniser, device=device)
        sparsegpt_pruner = SparseGPTWithCausalMasking(model, causal_calc)
        return sparsegpt_pruner.prune_with_causal_masking(inputs, sparsity)

    def _apply_magnitude_pruning(
        self, model, tokeniser, inputs, sparsity, device
    ):
        """Apply magnitude-based pruning."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                scores = torch.abs(module.weight.data)
                threshold = torch.quantile(scores.flatten(), sparsity)
                module.weight.data *= (scores > threshold).float()
        return model

    def _apply_gradient_pruning(
        self, model, tokeniser, inputs, sparsity, device
    ):
        """Apply gradient-based pruning."""
        model.train()
        outputs = model(**inputs)
        if hasattr(outputs, 'logits'):
            loss = outputs.logits.sum()
        else:
            loss = outputs.last_hidden_state.sum()
        loss.backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                scores = torch.abs(module.weight.grad)
                threshold = torch.quantile(scores.flatten(), sparsity)
                module.weight.data *= (scores > threshold).float()
        model.zero_grad(set_to_none=True)
        model.eval()
        return model

    def _compile_results(self) -> Dict[str, Any]:
        """Compile all experimental results."""
        results_df = pd.DataFrame(self.results)
        statistical_summary, performance_summary, viz_paths = {}, {}, {}
        if not results_df.empty:
            analyser = StatisticalAnalyser(results_df)
            statistical_summary = analyser.generate_statistical_summary()
            performance_summary = self._generate_performance_summary(results_df)
            viz_paths = self._generate_visualisations(results_df)
        return {
            "raw_results": results_df,
            "statistical_summary": statistical_summary,
            "performance_summary": performance_summary,
            "visualisation_paths": viz_paths,
            "experiment_config": self.config,
            "total_experiments": len(results_df),
            "completion_time": pd.Timestamp.now()
        } 

    def _generate_performance_summary(
        self, results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        if results_df.empty:
            return {}
        
        summary = {}
        agg_funcs = ["mean", "std", "median", "min", "max", "count"]
        
        summary["method_performance"] = results_df.groupby(
            "pruning_method")["performance"].agg(agg_funcs).round(4).to_dict()
        summary["sparsity_performance"] = results_df.groupby(
            "sparsity")["performance"].agg(agg_funcs).round(4).to_dict()
        summary["task_performance"] = results_df.groupby(
            "task_type")["performance"].agg(agg_funcs).round(4).to_dict()
        summary["model_performance"] = results_df.groupby(
            "model")["performance"].agg(agg_funcs).round(4).to_dict()
        
        high_sparsity_df = results_df[results_df["sparsity"] >= 0.6]
        if not high_sparsity_df.empty:
            summary["high_sparsity_performance"] = high_sparsity_df.groupby(
                "pruning_method"
            )["performance"].agg(["mean", "std", "count"]).round(4).to_dict()
            
        return summary

    def _setup_academic_style(self, results_df: pd.DataFrame):
        """Sets up matplotlib and seaborn styles for academic publication."""
        sns.set_style("ticks")
        # Use a colorblind-friendly palette for accessibility and better printing
        try:
            # Determine the number of unique methods to set the palette size
            n_methods = results_df["pruning_method"].nunique()
            palette = sns.color_palette("colorblind", n_methods)
        except:
            palette = sns.color_palette("colorblind")
        
        # Font settings (customize as needed)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        return palette

    def _plot_performance_vs_sparsity(self, results_df: pd.DataFrame, palette) -> str:
        """Figure A: Performance vs Sparsity with 95% CI."""
        plt.figure(figsize=(10, 6))

        # Define markers for better differentiation (e.g., in black/white printing)
        methods = results_df["pruning_method"].unique()
        markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
        # Ensure enough markers for all methods
        if len(methods) > len(markers):
            markers = markers * (len(methods) // len(markers) + 1)
        marker_map = {method: markers[i] for i, method in enumerate(methods)}

        # Use lineplot with error bands (95% CI)
        sns.lineplot(
            data=results_df, x="sparsity", y="performance", hue="pruning_method",
            style="pruning_method", markers=marker_map, dashes=False, palette=palette,
            errorbar=('ci', 95), linewidth=2, markersize=8
        )

        # Add baseline reference line (Sparsity=0.0)
        if 0.0 in results_df["sparsity"].values:
            baseline_perf = results_df[results_df["sparsity"] == 0.0]["performance"].mean()
            plt.axhline(baseline_perf, color='gray', linestyle='--', alpha=0.7, label=f'Baseline Avg: {baseline_perf:.3f}')

        #plt.yscale('log') 
        plt.title("Performance vs. Sparsity Level (95% CI)")
        plt.xlabel("Sparsity")
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.ylabel("Average Performance")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine() # Remove the top and right spines

        path = f"{self.config.results_dir}/FigA_performance_vs_sparsity.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def _plot_task_robustness(self, results_df: pd.DataFrame, palette) -> str:
        """Figure B: Performance drop at high sparsity across task types."""
        
        # Determine the highest sparsity level tested
        max_sparsity = results_df["sparsity"].max()
        if max_sparsity == 0.0:
            return ""

        # 1. Calculate baseline performance (sparsity=0.0)
        baseline_df = results_df[results_df["sparsity"] == 0.0]
        # Calculate average baseline performance per model/dataset/task across runs
        baseline_perf = baseline_df.groupby(["model", "dataset", "task_type"])["performance"].mean().reset_index()
        baseline_perf.rename(columns={"performance": "baseline_performance"}, inplace=True)

        # 2. Filter data for the high sparsity level
        # Use near equality for floating point comparison
        high_sparsity_df = results_df[abs(results_df["sparsity"] - max_sparsity) < 1e-5]

        # 3. Merge and calculate performance drop
        merged_df = pd.merge(high_sparsity_df, baseline_perf, on=["model", "dataset", "task_type"], how='left')
        merged_df = merged_df.dropna(subset=["baseline_performance"])
        # Avoid division by zero if baseline performance is 0
        merged_df["baseline_performance_safe"] = merged_df["baseline_performance"].replace(0, 1e-8)

        # Calculate drop percentage relative to baseline
        merged_df["performance_drop_pct"] = (
            (merged_df["baseline_performance"] - merged_df["performance"]) /
            merged_df["baseline_performance_safe"]
        ) * 100

        # 4. Plotting (Using barplot to compare means with CI)
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=merged_df,
            x="task_type",
            y="performance_drop_pct",
            hue="pruning_method",
            palette=palette,
            errorbar=('ci', 95),
            capsize=.1
        )

        plt.title(f"Performance Drop Rate at {max_sparsity*100:.0f}% Sparsity by Task Type")
        plt.xlabel("Task Type")
        plt.ylabel("Performance Drop (%)")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.axhline(0, color='grey', linestyle='--', alpha=0.7) # Line at 0% drop
        sns.despine()

        path = f"{self.config.results_dir}/FigB_task_robustness_analysis.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def _plot_causal_importance_distribution(self) -> str:
        """Figure C: Visualise the distribution of causal importance across layers."""
        if not self.importance_cache:
            return ""

        # 1. Aggregate scores from the cache
        score_data = []
        for (model_name, dataset_name), scores in self.importance_cache.items():
            if "causal" in scores:
                # Find task type associated with the dataset from configuration
                task_type = "Unknown"
                for cfg in self.config.datasets:
                    if cfg.name == dataset_name:
                        task_type = cfg.task_type
                        break
                
                for layer_name, importance in scores["causal"].items():
                    # Attempt to extract layer index (Heuristic for BERT-like architectures)
                    try:
                        # e.g., "encoder.layer.5.attention..."
                        parts = layer_name.split('.')
                        if 'layer' in parts:
                            layer_index = int(parts[parts.index('layer') + 1])
                        else:
                            continue # Skip non-layer specific modules (e.g., pooler)
                    except (ValueError, IndexError):
                        continue

                    score_data.append({
                        "model": model_name,
                        "task_type": task_type,
                        "layer_index": layer_index,
                        "importance": importance.item() if torch.is_tensor(importance) else importance
                    })

        if not score_data:
            return ""

        plot_df = pd.DataFrame(score_data)

        # 2. Plotting (Averaged by Task Type)
        plt.figure(figsize=(10, 6))

        # Show how importance distribution varies by the type of task across all models
        sns.lineplot(
            data=plot_df,
            x="layer_index",
            y="importance",
            hue="task_type",
            style="task_type",
            marker='o',
            errorbar=('ci', 95),
            linewidth=2
        )

        plt.title("Average Causal Importance Distribution Across Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("Causal Importance (KL Divergence)")
        plt.legend(title="Task Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine()

        path = f"{self.config.results_dir}/FigC_causal_importance_distribution.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    
    def _plot_actual_vs_target_sparsity(self, results_df: pd.DataFrame, palette) -> str:
        """Figure D: Actual vs. Target Sparsity to show the effect of causal protection."""
        plt.figure(figsize=(10, 6))

        # Ensure the required columns exist
        if "target_sparsity" not in results_df.columns or "actual_sparsity" not in results_df.columns:
            logger.warning("Skipping 'Actual vs Target Sparsity' plot: required columns not found.")
            return ""

        # Define markers for better differentiation, consistent with other plots
        methods = results_df["pruning_method"].unique()
        markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
        if len(methods) > len(markers):
            markers = markers * (len(methods) // len(markers) + 1)
        marker_map = {method: markers[i] for i, method in enumerate(methods)}

        # Main plot showing the relationship for each method
        sns.lineplot(
            data=results_df, 
            x="target_sparsity", 
            y="actual_sparsity", 
            hue="pruning_method",
            style="pruning_method", 
            markers=marker_map, 
            dashes=False, 
            palette=palette,
            errorbar=('ci', 95), 
            linewidth=2, 
            markersize=8
        )

        # Add a y=x diagonal reference line
        # This line represents the ideal case where actual sparsity equals target sparsity
        plt.axline([0, 0], [1, 1], color='red', linestyle='--', label='Ideal (y=x)')

        # Formatting the plot for clarity
        plt.title("Actual vs. Target Sparsity")
        plt.xlabel("Target Sparsity")
        plt.ylabel("Actual Measured Sparsity")
        
        # Format both axes as percentages for intuitive reading
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        
        # Ensure the plot limits are from 0 to 1 (or 0% to 100%)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine()

        # Save the figure
        path = f"{self.config.results_dir}/FigD_actual_vs_target_sparsity.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def _generate_visualisations(
        self, results_df: pd.DataFrame
    ) -> Dict[str, str]:
        """Generate and save comprehensive visualisations."""
        if results_df.empty:
            return {}

        viz_paths = {}
        # Apply academic style settings and get the color palette
        palette = self._setup_academic_style(results_df)

        logger.info("Generating academic visualisations...")

        # --- Figure A: Performance vs Sparsity ---
        try:
            path = self._plot_performance_vs_sparsity(results_df, palette)
            if path: viz_paths["FigA_sparsity_analysis"] = path
        except Exception as e:
            logger.warning(f"Failed to generate Figure A (Sparsity Analysis): {e}")

        # --- Figure B: Task Robustness Analysis ---
        try:
            path = self._plot_task_robustness(results_df, palette)
            if path: viz_paths["FigB_task_robustness"] = path
        except Exception as e:
            logger.warning(f"Failed to generate Figure B (Task Robustness): {e}")

        # --- Figure C: Causal Importance Distribution ---
        try:
            # Figure C uses its own palette logic based on task types, not methods
            path = self._plot_causal_importance_distribution()
            if path: viz_paths["FigC_causal_distribution"] = path
        except Exception as e:
            logger.warning(f"Failed to generate Figure C (Causal Distribution): {e}")

        # --- Figure D: Actual vs Target Sparsity ---
        try:
            path = self._plot_actual_vs_target_sparsity(results_df, palette)
            if path: viz_paths["FigD_actual_vs_target"] = path
        except Exception as e:
            logger.warning(f"Failed to generate Figure D (Actual vs Target Sparsity): {e}")

        # Reset style to default after generation
        plt.style.use('default')
        return viz_paths

    def _save_intermediate_results(self):
        """Save intermediate results to a CSV file."""
        if self.results:
            path = f"{self.config.results_dir}/intermediate_results.csv"
            pd.DataFrame(self.results).to_csv(path, index=False)
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save all final results, summaries, and reports."""
        results_df = results["raw_results"]
        if not results_df.empty:
            results_df.to_csv(
                f"{self.config.results_dir}/comprehensive_results.csv",
                index=False
            )
        
        json_results = {
            "statistical_summary": results["statistical_summary"],
            "performance_summary": results["performance_summary"],
            "total_experiments": results["total_experiments"],
            "completion_time": str(results["completion_time"])
        }
        with open(f"{self.config.results_dir}/comprehensive_results.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        self._generate_markdown_report(results)
        logger.info(f"Results saved to {self.config.results_dir}")
    

    def _generate_markdown_report(self, results: Dict[str, Any]):
        """Generate a comprehensive markdown report of the results."""
        report_path = f"{self.config.results_dir}/summary_report.md"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Causal Pruning Validation Report\n\n")
                f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Experiments:** {results.get('total_experiments', 0)}\n\n")

                # --- Figure D: Conceptual Framework (Mermaid Diagram) ---
                f.write("## Conceptual Framework (Figure D)\n\n")
                f.write("```mermaid\n")
                f.write("graph TD\n")
                f.write("    subgraph A[A: Causal Importance Estimation]\n")
                f.write("        direction LR\n")
                f.write("        Input[Calibration Data] --> Model\n")
                f.write("        Intervention[Intervention: Noise Injection at Layer L] --> Model\n")
                f.write("        Model -- Original Output --> O1[P(Y|X)]\n")
                f.write("        Model -- Intervened Output --> O2[P(Y|X, do(L=noise))]\n")
                f.write("        O1 & O2 --> KLD[KL Divergence]\n")
                f.write("        KLD --> Map[Causal Importance Map]\n")
                f.write("    end\n\n")
                f.write("    subgraph B[B: Causal Adaptive Pruning]\n")
                f.write("        direction LR\n")
                f.write("        A2[Input Data] --> B2(Calculate Scores - Wanda/SparseGPT);\n")
                f.write("        Map2[Causal Importance Map] --> C2(Adaptive Masking/Hierarchical Selection);\n")
                f.write("        B2 --> C2\n")
                f.write("        C2 --> D2{Apply Threshold};\n")
                f.write("        D2 --> E2[Pruned Model];\n")
                f.write("    end\n")
                f.write("```\n\n")

                # --- Table E: Statistical Significance Analysis ---
                f.write("## Statistical Significance Analysis (Table E)\n\n")
                f.write("Wilcoxon signed-rank test results compared against the baseline (default: Magnitude):\n\n")

                stats_summary = results.get("statistical_summary", {}).get("significance_tests", {})
                if stats_summary:
                    headers = ["Method (A)", "Baseline (B)", "Mean Diff (A-B)", "p-value", "Effect Size (Cohen's d)", "Significant?"]
                    table_data = []
                    
                    # Assuming baseline is 'Magnitude' based on StatisticalAnalyser implementation
                    baseline_name = "Magnitude" 

                    for method, data in stats_summary.items():
                        p_val = data.get("p_value", 1.0)
                        effect_size = data.get("effect_size", 0.0)
                        mean_diff = data.get("mean_difference", 0.0)
                        significant = data.get("significant", False)

                        # Formatting p-value with significance markers
                        if p_val < 0.001: p_str = "**<0.001***"
                        elif p_val < 0.01: p_str = f"**{p_val:.3f}**"
                        elif p_val < 0.05: p_str = f"**{p_val:.3f}*"
                        else: p_str = f"{p_val:.3f}"

                        table_data.append([
                            method,
                            baseline_name,
                            f"{mean_diff:+.4f}",
                            p_str,
                            f"{effect_size:.3f}",
                            'Yes' if significant else 'No'
                        ])
                    
                    # Use tabulate for clean Markdown table generation
                    f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
                    f.write("\n\n(* p<0.05, ** p<0.01, *** p<0.001)\n\n")
                else:
                    f.write("Statistical summary not available.\n\n")

                # --- Visualisations Links/Embedding ---
                f.write("## Visualisations\n\n")
                viz_paths = results.get("visualisation_paths", {})
                if viz_paths:
                    # Use Path().name to get relative paths for embedding images in Markdown
                    if 'FigA_sparsity_analysis' in viz_paths and viz_paths['FigA_sparsity_analysis']:
                        f.write(f"### Figure A: Performance vs. Sparsity\n")
                        f.write(f"![Figure A]({Path(viz_paths['FigA_sparsity_analysis']).name})\n\n")
                    if 'FigB_task_robustness' in viz_paths and viz_paths['FigB_task_robustness']:
                        f.write(f"### Figure B: Task Robustness Analysis\n")
                        f.write(f"![Figure B]({Path(viz_paths['FigB_task_robustness']).name})\n\n")
                    if 'FigC_causal_distribution' in viz_paths and viz_paths['FigC_causal_distribution']:
                        f.write(f"### Figure C: Causal Importance Distribution\n")
                        f.write(f"![Figure C]({Path(viz_paths['FigC_causal_distribution']).name})\n\n")

            logger.info(f"Summary report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error generating Markdown report: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("This script defines the configuration and framework.")
    logger.info("To run experiments, please use 'casual_pruning_execution.py'.")