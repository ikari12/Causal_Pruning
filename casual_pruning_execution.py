#!/usr/-bin/env python3
"""
Causal Pruning Execution Script
==============================

This script provides different execution modes for the comprehensive causal
pruning framework.
"""
# Standard library imports
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
import shutil

# Third-party imports
import pandas as pd
from tabulate import tabulate
import matplotlib.ticker as mtick

# Local application imports
from casual_pruning_config import (
    ComprehensiveValidationFramework,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    PruningConfig,
    get_jmteb_datasets,
    get_mteb_datasets,
    get_model_configs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_demo_config() -> ExperimentConfig:
    """
    Create a focused demo configuration for quick validation.
    """
    models = [
        ModelConfig(
            name="Ruri-V2-Base",
            model_path="cl-nagoya/ruri-base-v2",
            tokenizer_path="cl-nagoya/ruri-base-v2",
            max_length=512
        )
    ]
    datasets = [
        DatasetConfig(
            name="JMTEB-JSTS",
            dataset_path="sbintuitions/JMTEB",
            subset="jsts",
            task_type="similarity",
            metric="pearson",
            text_columns=["sentence1", "sentence2"],
            label_column="label",
        ),
    ]
    # Correction: Add pure 'Wanda' and 'SparseGPT' to the list.
    pruning_configs = [
        PruningConfig(
            method_name="Causal",
            sparsity_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        PruningConfig(
            method_name="Wanda",
            sparsity_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        PruningConfig(
            method_name="SparseGPT",
            sparsity_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        PruningConfig(
            method_name="HierarchicalCausalWanda",
            sparsity_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        PruningConfig(
            method_name="HierarchicalCausalSparseGPT",
            sparsity_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        PruningConfig(
            method_name="Gradient",
            sparsity_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        PruningConfig(
            method_name="Magnitude",
            sparsity_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
    ]
    return ExperimentConfig(
        models=models,
        datasets=datasets,
        pruning_configs=pruning_configs,
        random_seed=42,
        num_runs=1,
        statistical_test="wilcoxon",
        significance_level=0.05,
        results_dir="/app/results",
        save_intermediate=True
    )


def create_standard_config(
    num_models: int = 4, num_datasets: int = 15
) -> ExperimentConfig:
    """Create standard configuration for balanced evaluation."""
    all_models = get_model_configs()
    models = all_models[:num_models]

    jmteb_datasets = get_jmteb_datasets()[:num_datasets // 2]
    mteb_datasets = get_mteb_datasets()[:num_datasets // 2]
    datasets = jmteb_datasets + mteb_datasets
    for dataset in datasets:
        dataset.num_samples = 300

    pruning_configs = [
        PruningConfig(
            method_name="Causal",
            sparsity_levels=[0.0, 0.2, 0.4, 0.6, 0.8]
        ),
        PruningConfig(
            method_name="CausalMaskedWanda",
            sparsity_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
            causal_masking=True
        ),
        PruningConfig(
            method_name="CausalMaskedSparseGPT",
            sparsity_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
            causal_masking=True
        ),
        PruningConfig(
            method_name="Magnitude",
            sparsity_levels=[0.0, 0.2, 0.4, 0.6, 0.8]
        ),
        PruningConfig(
            method_name="Gradient",
            sparsity_levels=[0.0, 0.2, 0.4, 0.6, 0.8]
        ),
    ]

    # Correction: The 'return' statement was missing in the previous version.
    return ExperimentConfig(
        models=models,
        datasets=datasets,
        pruning_configs=pruning_configs,
        random_seed=42,
        num_runs=3,
        statistical_test="wilcoxon",
        significance_level=0.05,
        results_dir="standard_causal_pruning_results",
        save_intermediate=True
    )


def create_full_config() -> ExperimentConfig:
    """Create full configuration for complete JMTEB and MTEB evaluation."""
    models = get_model_configs()
    datasets = get_jmteb_datasets() + get_mteb_datasets()

    for dataset in datasets:
        if dataset.task_type in ["classification", "pair_classification"]:
            dataset.num_samples = 500
        elif dataset.task_type == "similarity":
            dataset.num_samples = 1000
        elif dataset.task_type in ["clustering", "retrieval", "reranking"]:
            dataset.num_samples = 300

    sparsity_levels = [i / 10.0 for i in range(10)]
    pruning_configs = [
        PruningConfig(method_name="Causal", sparsity_levels=sparsity_levels),
        PruningConfig(
            method_name="CausalMaskedWanda",
            sparsity_levels=sparsity_levels,
            causal_masking=True
        ),
        PruningConfig(
            method_name="CausalMaskedSparseGPT",
            sparsity_levels=sparsity_levels,
            causal_masking=True
        ),
        PruningConfig(method_name="Magnitude", sparsity_levels=sparsity_levels),
        PruningConfig(method_name="Gradient", sparsity_levels=sparsity_levels),
    ]

    # Correction: The 'return' statement was missing in the previous version.
    return ExperimentConfig(
        models=models,
        datasets=datasets,
        pruning_configs=pruning_configs,
        random_seed=42,
        num_runs=5,
        statistical_test="wilcoxon",
        significance_level=0.05,
        results_dir="full_causal_pruning_results",
        save_intermediate=True
    )


def load_custom_config(config_path: str) -> ExperimentConfig:
    """Load custom configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    models = [ModelConfig(**md) for md in config_dict.get("models", [])]
    datasets = [DatasetConfig(**dd) for dd in config_dict.get("datasets", [])]
    pruning_configs = [
        PruningConfig(**pd) for pd in config_dict.get("pruning_configs", [])
    ]
    experiment_dict = config_dict.get("experiment", {})

    return ExperimentConfig(
        models=models,
        datasets=datasets,
        pruning_configs=pruning_configs,
        **experiment_dict
    )


def save_config_template(path: str):
    """Save a template configuration file."""
    template = {
        "models": [{
            "name": "BERT-Base-Uncased", "model_path": "bert-base-uncased",
            "tokenizer_path": "bert-base-uncased", "max_length": 512,
            "device": "auto"
        }],
        "datasets": [{
            "name": "MTEB-ImdbClassification", "dataset_path": "mteb/imdb",
            "task_type": "classification", "metric": "accuracy",
            "text_columns": ["text"], "label_column": "label",
            "num_samples": 100
        }],
        "pruning_configs": [{
            "method_name": "Causal", "sparsity_levels": [0.0, 0.4, 0.8],
            "causal_masking": False
        }],
        "experiment": {
            "random_seed": 42, "num_runs": 3,
            "statistical_test": "wilcoxon", "significance_level": 0.05,
            "results_dir": "custom_results", "save_intermediate": True
        }
    }
    with open(path, 'w') as f:
        json.dump(template, f, indent=2)
    logger.info(f"Configuration template saved to {path}")


def estimate_runtime(config: ExperimentConfig) -> Dict[str, Any]:
    """Provide a rough estimate of the total runtime."""
    if not config.pruning_configs:
        return {"total_experiments": 0, "estimated_time_str": "0h 0m"}

    num_configs = (
        len(config.models) *
        len(config.datasets) *
        len(config.pruning_configs) *
        len(config.pruning_configs[0].sparsity_levels)
    )
    total_experiments = num_configs * config.num_runs

    time_per_task = {
        "classification": 2, "similarity": 3, "clustering": 5,
        "retrieval": 8, "reranking": 6, "pair_classification": 3
    }

    avg_task_time = sum(
        time_per_task.get(ds.task_type, 3) for ds in config.datasets
    ) / len(config.datasets) if config.datasets else 3

    total_minutes = avg_task_time * num_configs * config.num_runs
    hours, minutes = divmod(total_minutes, 60)

    return {
        "total_experiments": total_experiments,
        "estimated_time_str": f"{int(hours)}h {int(minutes)}m"
    }


def print_configuration_summary(config: ExperimentConfig):
    """Print a summary of the experiment configuration."""
    logger.info("=" * 80)
    logger.info("CAUSAL PRUNING VALIDATION CONFIGURATION")
    logger.info("=" * 80)

    logger.info(f"Models ({len(config.models)}):")
    for model in config.models:
        logger.info(f"  • {model.name}")

    logger.info(f"\nDatasets ({len(config.datasets)}):")
    task_counts = {}
    for dataset in config.datasets: # This line is now correct
        task_counts[dataset.task_type] = (
            task_counts.get(dataset.task_type, 0) + 1
        )
    for task_type, count in task_counts.items():
        logger.info(f"  • {task_type}: {count} tasks")

    logger.info(f"\nPruning Methods ({len(config.pruning_configs)}):")
    for pruning in config.pruning_configs:
        sparsity_range = (
            f"{min(pruning.sparsity_levels):.1f}-"
            f"{max(pruning.sparsity_levels):.1f}"
        )
        logger.info(f"  • {pruning.method_name} (sparsity: {sparsity_range})")

    logger.info("\nExperimental Parameters:")
    logger.info(f"  • Random seed: {config.random_seed}")
    logger.info(f"  • Runs per experiment: {config.num_runs}")
    logger.info(f"  • Statistical test: {config.statistical_test}")
    logger.info(f"  • Results directory: {config.results_dir}")

    runtime_est = estimate_runtime(config)
    logger.info("\nRuntime Estimation:")
    logger.info(f"  • Total experiments: {runtime_est['total_experiments']:,}")
    logger.info(f"  • Estimated time: ~{runtime_est['estimated_time_str']}")
    logger.info("=" * 80)


def main():
    """Parse arguments and run the validation framework."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Causal Pruning Validation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=["demo", "standard", "full", "custom"],
        default="demo", help="Execution mode (default: demo)"
    )
    parser.add_argument(
        "--models", type=int, default=4,
        help="Number of models for standard mode (default: 4)"
    )
    parser.add_argument(
        "--datasets", type=int, default=15,
        help="Number of datasets for standard mode (default: 15)"
    )
    parser.add_argument(
        "--config", type=str, help="Path to custom configuration JSON file"
    )
    parser.add_argument(
        "--template", type=str,
        help="Generate a configuration template at the specified path"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration and estimates without running experiments"
    )
    parser.add_argument(
        "--resume", type=str,
        help="Resume from an intermediate results directory"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete the importance score cache before running"
    )
    args = parser.parse_args()

    if args.template:
        save_config_template(args.template)
        return

    config = None
    if args.mode == "demo":
        config = create_demo_config()
    elif args.mode == "standard":
        config = create_standard_config(args.models, args.datasets)
    elif args.mode == "full":
        config = create_full_config()
    elif args.mode == "custom":
        if not args.config:
            logger.error("Custom mode requires the --config parameter.")
            sys.exit(1)
        config = load_custom_config(args.config)

    if config is None:
        logger.error(f"Invalid mode '{args.mode}' selected.")
        sys.exit(1)

    print_configuration_summary(config)

    if args.clear_cache:
        cache_dir = Path(config.results_dir) / "importance_cache"
        if cache_dir.exists():
            logger.info(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        else:
            logger.info("Cache directory not found, nothing to clear.")

    if args.dry_run:
        logger.info("Dry run completed. No experiments were run.")
        return

    try:
        framework = ComprehensiveValidationFramework(config)
        if args.resume:
            intermediate_path = Path(args.resume) / "intermediate_results.csv"
            if intermediate_path.exists():
                framework.results = pd.read_csv(
                    intermediate_path
                ).to_dict('records')
                logger.info(
                    f"Resumed with {len(framework.results)} existing results"
                )

        logger.info("Starting comprehensive validation...")
        framework.run_comprehensive_validation()

        logger.info("=" * 80)
        logger.info("COMPREHENSIVE VALIDATION COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {config.results_dir}")

    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()