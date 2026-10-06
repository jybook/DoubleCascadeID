"""Apply trained model to new sqlite files"""

import os
from typing import Any, Dict, List, Optional
import glob as glob

from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
PARENT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/'
EXAMPLE_OUTPUT_DIR = PARENT_DIR + 'training_output'

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.custom_tasks import DoubleCascadeReconstruction
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataloader import DataLoader

# data_parent_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/HNL/sqlite2/'
data_parent_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/HNL/sqlite_decay_constraint/'
nue_sets = ['oscNext_genie_level7_v02.00_pass2.121122.000291.db',
             'oscNext_genie_level7_v02.00_pass2.121122.000292.db',
             'oscNext_genie_level7_v02.00_pass2.121122.000293.db',
             'oscNext_genie_level7_v02.00_pass2.121122.000294.db',
             'oscNext_genie_level7_v02.00_pass2.121122.000295.db',]
numu_sets = ['oscNext_genie_level7_v02.00_pass2.141122.000291.db',
             'oscNext_genie_level7_v02.00_pass2.141122.000292.db',
             'oscNext_genie_level7_v02.00_pass2.141122.000293.db',
             'oscNext_genie_level7_v02.00_pass2.141122.000294.db',
             'oscNext_genie_level7_v02.00_pass2.141122.000295.db',]
nutau_sets = ['oscNext_genie_level7_v02.00_pass2.161122.000291.db',
             'oscNext_genie_level7_v02.00_pass2.161122.000292.db',
             'oscNext_genie_level7_v02.00_pass2.161122.000293.db',
             'oscNext_genie_level7_v02.00_pass2.161122.000294.db',
             'oscNext_genie_level7_v02.00_pass2.161122.000295.db']

for superset in zip(nue_sets, nutau_sets, numu_sets):
    for set in superset:
        set = data_parent_dir+'oscNext/'+set
        print(set)

dataset_ids = ['191303']#['oscNext']#
# Include oscNext for background, exclude for signal
dataset_paths = []
for did in dataset_ids:
    dataset_paths.extend(glob.glob(os.path.join(data_parent_dir, "{}".format(did), "merged/merged.db")))

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.remove("interaction_time")
truth.append('dbang_decay_length')
truth.append('event_weight')

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
            project="double_cascade_estimator_quality_cut",
            # entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Choose the dataset you want HERE
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
    }

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "background_verification")
    run_name = "dynedge_{}".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Define graph representation
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
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "drop_last": True,
        },
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "drop_last": True,
        },
        validation_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "drop_last": False,
        },
    )

    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader

    # Building model

    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = DoubleCascadeReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.where(x>0.1,torch.log10(x),-1),
        transform_inference=lambda x: torch.pow(10, x),
    )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-02},#Raised eps from 1e-3
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

    ################# NEW
        
    # Save predictions to file
    db_name = "deepcore_cut" 
    # path = os.path.join(archive, db_name, run_name)
    path = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/training_output/new_MC/deepcore_cut/dynedge_dbang_decay_length'
    logger.info(f"Reading model state from {path}")
    model.load_state_dict(f"{path}/reco_state_dict.pth")

    # Get predictions
    additional_attributes = model.target_labels
    additional_attributes.extend([
        #Truth level variables
        'pid', 'energy', 'event_weight', 'azimuth', 'zenith', 'position_x', 'position_y', 'position_z', 'elasticity', 'interaction_type', 'inelasticity', 
        #Event Features
        'dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area'])
    assert isinstance(additional_attributes, list)  # mypy
    print("additional attributes are", additional_attributes)

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config["fit"]["gpus"],
    )
    
    # Save predictions and model to file
    logger.info(f"Writing results to {path}")

    # Save results as .csv
    results.to_csv(f"{path}/100_epoch_hnl_verification_checks.csv")

    # results.to_csv(f"{path}/100_epoch_background_checks.csv")

    
if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Test GNN model which estimates decay lengths.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"{EXAMPLE_DATA_DIR}/merged.db" 
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
        default='dbang_decay_length', #"energy",#
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth", #"mc_truth",#
    )

    parser.with_standard_arguments(
        ("gpus", torch.cuda.device_count()),
        ("max-epochs", 100),
        ("early-stopping-patience", 5),
        ("batch-size", 15),
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
