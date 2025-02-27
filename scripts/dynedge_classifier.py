"""
Modified GraphNET example: Multi-class classification using DynEdge from pre-defined config files.
Modified starting 12/17/2024 by J. Book
"""

import os
from typing import List, Optional, Dict, Any

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from graphnet.data.dataset.dataset import EnsembleDataset
# from graphnet.constants import (
#     EXAMPLE_OUTPUT_DIR,
#     DATASETS_CONFIG_DIR,
#     MODEL_CONFIG_DIR,
# )
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.models import StandardModel
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import Logger

MC_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/HNL/parquet/'
SIGNAL_DIRS = ['190301', '190302', '190303', '190304', '190305', '190306', '190307', '190308']
BACKGROUND_DIRS = ['oscNext']
#Both of these have sub-folders 'SRTInIcePulses' and 'truth'

PARENT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/'
MODEL_CONFIG_DIR = PARENT_DIR + 'model_configs'
DATASETS_CONFIG_DIR = PARENT_DIR + 'dataset_configs/'
OUTPUT_DIR = PARENT_DIR + 'training_output'

def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    suffix: Optional[str] = None,
    wandb: bool = False,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-classifier",
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )

    # Build model
    model_config = ModelConfig.load(model_config_path)
    model: StandardModel = StandardModel.from_config(model_config, trust=True)


    # Configuration
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )
    print(config.target)
    logger.info(f"model targets from config: {config.target}" )
    import torch
    print(torch.cuda.is_available)
    logger.info(f"torch.cuda.is_available? {torch.cuda.is_available()}")
    
    if suffix is not None:
        archive = os.path.join(OUTPUT_DIR, f"train_model_{suffix}")
    else:
        archive = os.path.join(OUTPUT_DIR, "train_model")
    run_name = "dynedge_{}_example".format("_".join(config.target))

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    datasets: Dict[str, Any] = Dataset.from_config(
        dataset_config,
    )

    # Construct datasets from multiple selections
    train_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("valid")]
    )
    test_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("test")]
    )
    print("\n *********************************")
    print(dataset_config)
    print(" *********************************\n")
    # Construct dataloaders
    train_dataloaders = DataLoader(
        train_dataset, shuffle=True, **config.dataloader
    )
    valid_dataloaders = DataLoader(
        valid_dataset, shuffle=False, **config.dataloader
    )
    test_dataloaders = DataLoader(
        test_dataset, shuffle=False, **config.dataloader
    )
    print("\n Dataloaders:")
    print("train: ", next(iter(train_dataloaders)))
    print("valid: ", next(iter(valid_dataloaders)))
    print("test : ", next(iter(test_dataloaders)))

    # Log configurations to W&B
    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
    #     training.
    if wandb and rank_zero_only == 0:
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(model_config.as_dict())
        wandb_logger.experiment.config.update(dataset_config.as_dict())

    # Training model
    model.fit(
        train_dataloaders,
        valid_dataloaders,
        early_stopping_patience=config.early_stopping_patience,
        logger=wandb_logger if wandb else None,
        **config.fit,
    )

    # Save model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    os.makedirs(path, exist_ok=True)
    logger.info(f"Writing results to {path}")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")

    # Get predictions
    if isinstance(config.target, str):
        additional_attributes = [config.target]
    else:
        additional_attributes = config.target

    logger.info(f"config.target: {config.target}")
    logger.info(f"prediction_columns: {model.prediction_labels}")

    results = model.predict_as_dataframe(
        test_dataloaders,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config.fit["gpus"],
    )
    results.to_csv(f"{path}/dynedge_example_results.csv")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
            Train GNN classification model.
            """
    )

    parser.with_standard_arguments(
        (
            "dataset-config",
            os.path.join(
                DATASETS_CONFIG_DIR,
                "training_example_modified_parquet.yml",
                # "training_classification_example_data_sqlite.yml",
            ),
        ),
        (
            "model-config",
            os.path.join(
                MODEL_CONFIG_DIR, 
                "dynedge_PID_classification_modified.yml" 
                # "dynedge_PID_classification_example.yml"#
            ),
        ),
        "gpus",
        ("max-epochs", 1),
        "early-stopping-patience",
        ("batch-size", 16),
        ("num-workers",1),
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Name addition to folder (default: %(default)s)",
        default=None,
    )

    args, unknown = parser.parse_known_args()

    main(
        args.dataset_config,
        args.model_config,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.suffix,
    )
