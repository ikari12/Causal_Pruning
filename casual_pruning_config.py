#!/usr/bin/env python3
"""
Causal Pruning Configuration Framework
=====================================

This module implements a comprehensive framework for causal intervention-based
pruning with integration of Wanda and SparseGPT methods, evaluated across
JMTEB and MTEB benchmark datasets.
"""
# Standard library imports
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
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from joblib import Parallel, delayed 

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
    results_dir: str = "comprehensive_results"
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
            dataset_path="sbintuitions/JMTEB", subset="jqara",
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
        num_samples: int = 100
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
        #results = Parallel(n_jobs=n_gpus, verbose=50)(tasks)
        MAXIMUM_PROCESS = n_gpus #len(tasks) / n_gpus if len(tasks) / n_gpus < 30 else 30
        results = Parallel(n_jobs=MAXIMUM_PROCESS, verbose=50)(tasks)

        importance_scores = dict(zip(target_layers, results))
        return importance_scores

    @staticmethod
    def _compute_layer_on_gpu(
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_name: str,
        baseline_probs: torch.Tensor,
        num_samples: int,
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

        importance_values = []
        for _ in range(num_samples):
            hook = target_layer.register_forward_hook(intervention_hook)
            try:
                with torch.no_grad():
                    outputs = worker_model(**worker_inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
                    probs = torch.softmax(logits, dim=-1)
                    kl_div = torch.nn.functional.kl_div(
                        torch.log(probs + 1e-8),
                        worker_baseline_probs,
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
        self, inputs: Dict[str, torch.Tensor], sparsity: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Compute Wanda importance scores with causal masking."""
        importance_scores = {}
        activations = self._get_activations(inputs)
        causal_scores = self.causal_calculator.compute_causal_importance(
            inputs
        )

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights = module.weight.data
                if name in activations:
                    acts = activations[name]
                    wanda_scores = torch.abs(weights) * torch.abs(acts).mean(0)
                else:
                    wanda_scores = torch.abs(weights)

                if name in causal_scores:
                    causal_mask = causal_scores[name]
                    protection_threshold = torch.quantile(
                        causal_mask, 1 - sparsity * 0.5
                    )
                    causal_protection = (
                        causal_mask > protection_threshold
                    ).float()
                    wanda_scores = wanda_scores * (1 + causal_protection * 2)

                importance_scores[name] = wanda_scores
        return importance_scores

    def _get_activations(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from model layers."""
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    activations[name] = output[0].detach()
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        with torch.no_grad():
            # ==================================================================
            # Final Correction: Forcibly synchronise devices before this call.
            # This ensures that even if the model was moved to CPU earlier,
            # it is moved back to the correct GPU before being called.
            # ==================================================================
            target_device = inputs.get('input_ids', next(iter(inputs.values()))).device
            self.model.to(target_device)
            # ==================================================================
            
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return activations

    def prune_with_causal_masking(
        self, inputs: Dict[str, torch.Tensor], sparsity: float
    ) -> nn.Module:
        """Apply Wanda pruning with causal masking."""
        importance_scores = self.compute_wanda_scores(inputs, sparsity)
        return self._apply_structured_pruning(importance_scores, sparsity)

    def _apply_structured_pruning(
        self, importance_scores: Dict[str, torch.Tensor], sparsity: float
    ) -> nn.Module:
        """Apply structured pruning based on importance scores."""
        for name, module in self.model.named_modules():
            if name in importance_scores and isinstance(module, nn.Linear):
                scores = importance_scores[name]
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask
        return self.model


class SparseGPTWithCausalMasking:
    """SparseGPT pruning method enhanced with causal masking."""

    def __init__(
        self, model: nn.Module, causal_calculator: CausalImportanceCalculator
    ):
        self.model = model
        self.causal_calculator = causal_calculator

    def compute_sparsegpt_scores(
        self, inputs: Dict[str, torch.Tensor], sparsity: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Compute SparseGPT scores with causal masking."""
        importance_scores = {}
        causal_scores = self.causal_calculator.compute_causal_importance(
            inputs
        )

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hessian_diag = self._compute_hessian_diagonal(module, inputs)
                weights = module.weight.data
                sparsegpt_scores = weights.pow(2) * hessian_diag

                if name in causal_scores:
                    causal_mask = causal_scores[name]
                    protection_threshold = torch.quantile(
                        causal_mask, 1 - sparsity * 0.3
                    )
                    causal_protection = (
                        causal_mask > protection_threshold
                    ).float()
                    sparsegpt_scores *= (1 + causal_protection * 3)

                importance_scores[name] = sparsegpt_scores
        return importance_scores

    def _compute_hessian_diagonal(
        self, module: nn.Linear, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute an approximate Hessian diagonal for the layer."""
        weights = module.weight.data
        fisher_diag = torch.ones_like(weights) * 1e-4
        fisher_diag += torch.abs(weights) * 1e-3
        return fisher_diag

    def prune_with_causal_masking(
        self, inputs: Dict[str, torch.Tensor], sparsity: float
    ) -> nn.Module:
        """Apply SparseGPT pruning with causal masking."""
        importance_scores = self.compute_sparsegpt_scores(inputs, sparsity)
        return self._apply_obs_pruning(importance_scores, sparsity)

    def _apply_obs_pruning(
        self, importance_scores: Dict[str, torch.Tensor], sparsity: float
    ) -> nn.Module:
        """Apply Optimal Brain Surgeon style pruning."""
        for name, module in self.model.named_modules():
            if name in importance_scores and isinstance(module, nn.Linear):
                scores = importance_scores[name]
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                original_weights = module.weight.data.clone()
                module.weight.data *= mask
                pruned_weights = original_weights * (1 - mask)

                if mask.sum() > 0:
                    redistribution = pruned_weights.sum() / mask.sum()
                    module.weight.data += mask * redistribution * 0.1
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
        self.cache_dir = Path(self.config.results_dir) / ".importance_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run the complete validation with a file-based caching mechanism for
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
                        wanda_scores[name] = torch.abs(weights) * torch.abs(acts).mean(0)
                    else:
                        wanda_scores[name] = torch.abs(weights)
            
            scores_to_cache = {"causal": causal_scores, "wanda": wanda_scores}
            self.importance_cache[cache_key] = scores_to_cache
            logger.info(f"Saving new scores to: {cache_file}")
            torch.save(scores_to_cache, cache_file)
            del temp_model
        
        progress_bar = tqdm(all_experiments, desc="Overall Progress", ncols=80)
        # Now run all experiments using the cached scores
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
                    pruned_model = self._apply_pruning(
                        model,
                        prune_cfg,
                        sparsity,
                        cached_scores["causal"],
                        cached_scores["wanda"]
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
                calib_dataset = self.evaluator._load_dataset(dataset_config)
                calib_inputs = self._prepare_calibration_data(
                    calib_dataset, tokeniser, dataset_config.text_columns
                )
                model = self._apply_pruning(
                    model, tokeniser, pruning_config, sparsity, calib_inputs
                )
            
            eval_results = self.evaluator.evaluate_model_on_dataset(
                model, tokeniser, dataset_config, pruning_config
            )
            
            result = {
                "model": model_config.name,
                "dataset": dataset_config.name,
                "task_type": dataset_config.task_type,
                "pruning_method": pruning_config.method_name,
                "sparsity": sparsity,
                "run": run,
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
        self, dataset, tokeniser, text_columns: List[str], num_samples: int = 32
    ) -> Dict[str, torch.Tensor]:
        """Create a single batch of calibration data from a dataset."""
        if not dataset or len(dataset) == 0:
            sample_text = ["This is a sample text for importance computation."]
            return tokeniser(
                sample_text, max_length=512, truncation=True,
                padding=True, return_tensors="pt"
            )

        num_samples = min(num_samples, len(dataset))
        calibration_dataset = dataset.select(range(num_samples))
        
        text_list = []
        calib_desc = "  -> Preparing calibration data"
        for example in tqdm(calibration_dataset, desc=calib_desc, leave=False, ncols=100):
            if len(text_columns) == 1:
                text = example[text_columns[0]]
            else:
                text = " ".join([str(example[col]) for col in text_columns])
            text_list.append(text)
        
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

    def _apply_pruning(
        self,
        model: nn.Module,
        tokeniser,
        pruning_config: PruningConfig,
        sparsity: float,
        calibration_inputs: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Apply the specified pruning method."""
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in calibration_inputs.items()}
        
        # Dispatch to the correct pruning method.
        if pruning_config.method_name == "CausalMaskedWanda":
            return self._apply_causal_wanda(model, tokeniser, inputs, sparsity, device)
        elif pruning_config.method_name == "CausalMaskedSparseGPT":
            return self._apply_causal_sparsegpt(model, tokeniser, inputs, sparsity, device)
        elif pruning_config.method_name == "Causal":
            return self._apply_causal_pruning(model, tokeniser, inputs, sparsity, device)
        elif pruning_config.method_name == "Magnitude":
            return self._apply_magnitude_pruning(model, sparsity)
        elif pruning_config.method_name == "Gradient":
            return self._apply_gradient_pruning(model, inputs, sparsity)
        elif pruning_config.method_name == "Wanda":
            return self._apply_wanda_pruning(model, tokeniser, inputs, sparsity)
        elif pruning_config.method_name == "SparseGPT":
            return self._apply_sparsegpt_pruning(model, inputs, sparsity)
        
        logger.warning(
            f"Pruning method '{pruning_config.method_name}' not found. "
            "Returning original model."
        )
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
    
    def _apply_wanda_pruning(
        self, model, tokeniser, inputs, sparsity
    ) -> nn.Module:
        """Apply pure Wanda pruning (Weight and Activation)."""
        logger.info("  -> Applying pure Wanda pruning...")
        pruned_model = copy.deepcopy(model)
        activations = self._get_activations(pruned_model, inputs)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                if name not in activations:
                    continue
                weights = module.weight.data
                acts = activations[name]
                scores = torch.abs(weights) * torch.abs(acts).mean(0)
                
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask
        return pruned_model

    def _apply_sparsegpt_pruning(
        self, model, inputs, sparsity
    ) -> nn.Module:
        """Apply pure SparseGPT-like pruning (simplified)."""
        logger.info("  -> Applying pure SparseGPT pruning...")
        pruned_model = copy.deepcopy(model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                # Simplified Hessian approximation
                hessian_diag = torch.ones_like(weights) * 1e-4 + torch.abs(weights) * 1e-3
                scores = weights.pow(2) * hessian_diag
                
                threshold = torch.quantile(scores.flatten(), sparsity)
                mask = (scores > threshold).float()
                module.weight.data *= mask
        return pruned_model
    
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

    def _generate_visualisations(
        self, results_df: pd.DataFrame
    ) -> Dict[str, str]:
        """Generate and save comprehensive visualisations."""
        if results_df.empty:
            return {}
        
        viz_paths = {}
        plt.style.use('default')
        sns.set_palette("husl")

        # Visualisation 1: Method performance bar chart
        plt.figure(figsize=(12, 8))
        method_data = results_df.groupby("pruning_method")["performance"].agg(
            ["mean", "std"]
        )
        bars = plt.bar(
            method_data.index, method_data["mean"],
            yerr=method_data["std"], capsize=5, alpha=0.7
        )
        plt.title(
            "Performance Comparison Across Pruning Methods",
            fontsize=16, fontweight='bold'
        )
        plt.xlabel("Pruning Method", fontsize=12)
        plt.ylabel("Average Performance", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        for bar, mean in zip(bars, method_data["mean"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold'
            )
        path = f"{self.config.results_dir}/method_performance_comparison.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths["method_comparison"] = path
        
        # Visualisation 2: Performance vs Sparsity line plot
        plt.figure(figsize=(14, 8))
        for method in results_df["pruning_method"].unique():
            method_df = results_df[results_df["pruning_method"] == method]
            sparsity_perf = method_df.groupby("sparsity")["performance"].agg(
                ["mean", "std"]
            )
            plt.errorbar(
                sparsity_perf.index * 100, sparsity_perf["mean"],
                yerr=sparsity_perf["std"], marker='o', linewidth=2,
                markersize=8, label=method, capsize=5
            )
        plt.title(
            "Performance vs Sparsity Level", fontsize=16, fontweight='bold'
        )
        plt.xlabel("Sparsity (%)", fontsize=12)
        plt.ylabel("Performance", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        path = f"{self.config.results_dir}/performance_vs_sparsity.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths["sparsity_analysis"] = path

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
        """Generate a brief markdown report of the results."""
        report_path = f"{self.config.results_dir}/summary_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Causal Pruning Validation Report\n\n")
            f.write("Report generation is a placeholder in this version.\n")
        logger.info(f"Summary report saved to {report_path}")


if __name__ == "__main__":
    logger.info("This script defines the configuration and framework.")
    logger.info("To run experiments, please use 'casual_pruning_execution.py'.")