"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
PARENT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/'
EXAMPLE_DATA_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/HNL/sqlite/190307/merged' #PARENT_DIR + 'parquet' #'sqlite' 
EXAMPLE_OUTPUT_DIR = PARENT_DIR + 'training_output'

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
# from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset

# # Constants
# features = FEATURES.PROMETHEUS
# truth = TRUTH.PROMETHEUS
import os
import glob as glob
import numpy as np
data_parent_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/HNL/sqlite/'
dataset_ids = [190301, 190302, 190303, 190304, 190305, 190306, 190307, 190308, 'oscNext']
# dataset_ids = [190301]
# Include oscNext for background, exclude for signal
dataset_paths = []
for did in dataset_ids:
    dataset_paths.extend(glob.glob(os.path.join(data_parent_dir, "{}".format(did), "merged/merged.db")))
print("datasets: \n")
print(dataset_paths)

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.remove("interaction_time") #we just call it time, in future iterations, maybe replace with the correct variable name?

# Some code to help debug the "list index out of range" error (from Rasmus)
from torch_geometric.data import Data, Batch
from typing import List

def collate_fn_debug(examples: List[Data]) -> Batch:
    """
    Group a list of training examples into a batch.
    """

    # Filter out events with less than two rows/nodes/hits
    examples_filtered = [ex for ex in examples if ex.n_pulses >= 1]
    
    try:
        batch = Batch.from_data_list(examples_filtered)
    except IndexError as e:
        print('collate_fn_debug: --- caught empty batch ---')
        print(f'empty event_nos: {[ex["event_no"] for ex in examples]}')
        print(f'data paths: {[ex["dataset_path"] for ex in examples]}')
        raise e
    return batch

def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
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
            project="energy-reco-example",
            # entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Switch to using multiple datasets
    path = dataset_paths
    
    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        "dataset_reference": SQLiteDataset,
        # if path.endswith(".db")
        # else ParquetDataset,
    }

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "multiple_epochs")
    run_name = "dynedge_{}_example".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Define graph representation
    # graph_definition = KNNGraph(detector=Prometheus())
    graph_definition = KNNGraph(
        detector=IceCube86(), 
        input_feature_names=['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area'],
        )
    
    # Use GraphNetDataModule to load in data
    dm = GraphNeTDataModule(
        dataset_reference=config["dataset_reference"],
        dataset_args={
            "truth": truth,
            "truth_table": truth_table,
            "features": features,
            "graph_definition": graph_definition,
            "pulsemaps": [config["pulsemap"]],
            "path": config["path"],
            #"selection": np.ones(len(config["path"])) #To use an ensemble dataset, a selection is required, and none doesn't seem to do it.
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "drop_last": True, # skip incomplete final batches
            # "collate_fn": collate_fn_debug,
        },
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "drop_last": True, # skip incomplete final batches
            # "collate_fn": collate_fn_debug,
        },
        validation_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "drop_last": False, # skip incomplete final batches
            # "collate_fn": collate_fn_debug,
        },
    )

    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader
    # testing_dataloader = dm.test_dataloader

    # Building model

    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = EnergyReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.log10(x),
        transform_inference=lambda x: torch.pow(10, x),
    )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    # Training model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    # Get predictions
    additional_attributes = ['pid', 'energy'] # model.target_labels
    assert isinstance(additional_attributes, list)  # mypy
    print("additional attributes are", additional_attributes)

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config["fit"]["gpus"],
    )
    print("saved columns are: ", results.keys())
    
    # Save predictions and model to file
    db_name = "merged_files" #path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{path}/energy_results.csv")

    # Save full model (including weights) to .pth file - not version safe
    # Note: Models saved as .pth files in one version of graphnet
    #       may not be compatible with a different version of graphnet.
    model.save(f"{path}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{path}/energy_reco_state_dict.pth")
    model.save_config(f"{path}/energy_model_config.yml")

    
    
if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model without the use of config files.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"{EXAMPLE_DATA_DIR}/merged.db" #f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default='SRTInIcePulses', #"total",#
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default='energy', #"total_energy",#
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth", #"mc_truth",#
    )

    parser.with_standard_arguments(
        ("gpus", torch.cuda.device_count()),
        ("max-epochs", 10),
        ("early-stopping-patience", 2),
        ("batch-size", 50),
        "num-workers",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args, unknown = parser.parse_known_args()

    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.wandb,
    )
