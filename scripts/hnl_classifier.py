import os
import numpy as np
import glob as glob
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from pytorch_lightning.callbacks import ModelCheckpoint

# from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge#,GRIT
from graphnet.models.graphs import KNNGraph
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.utilities.imports import has_jammy_flows_package

from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.models import StandardModel
import torch

import wandb    
from pytorch_lightning.loggers import WandbLogger


def main(
    pulsemap: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    filename: str,
    # num_workers: int,
    wandb: bool = True,
) -> None:
    # filename = "filtered_dataset"

    # Set data and output directories
    PARENT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/'
    EXAMPLE_OUTPUT_DIR = PARENT_DIR + 'training_output'
    data_parent_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/HNL/sqlite_L3/'
    dataset_ids = ['new_190300', 'new_190312', 'new_190313', 'new_190314', 'with_cuts/oscNext']
    # Include oscNext for background, exclude for signal
    # Removing 190311 because it has a lot of hnl single cascade events

    dataset_paths = []
    for did in dataset_ids:
        dataset_paths.extend(glob.glob(os.path.join(data_parent_dir, "{}".format(did), "merged/merged.db")))
    print("datasets: \n")
    print(dataset_paths)

    # Constants
    features = FEATURES.ICECUBE86 # Nick adds hlc
    truth = TRUTH.ICECUBE86
    truth.remove("interaction_time") #we just call it time, in future iterations, maybe replace with the correct variable name?
    # truth.append('time')
    truth.append('dbang_decay_length')
    truth.append('event_weight')
    truth.append('is_hnl')
    print("Truth: ",truth)
    print("features: ", features)

    # Make sure the jammy flows is installed
    try:
        assert has_jammy_flows_package()
        from graphnet.models import NormalizingFlow
    except AssertionError:
        raise AssertionError(
            "This example requires the package`jammy_flow` "
            " to be installed. It appears that the package is "
            " not installed. Please install the package."
        )

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="hnl_classifier",
            # entity="julia-y-book-harvard-university",
            save_dir=wandb_dir,
            log_model=True,
        )

    # Configuration
    config: Dict[str, Any] = {
        "path": dataset_paths,
        "pulsemap": 'SRTInIcePulses',
        "batch_size": batch_size,
        "num_workers": 8,
        "target": 'pid',
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
            "distribution_strategy":"ddp_find_unused_parameters_true"
        },
        "dataset_reference": SQLiteDataset,
    }

    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    #### Start the physics ####

    # Define graph representation
    graph_definition = KNNGraph(
        detector=IceCube86(), 
        input_feature_names=features, #['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area'],
        nb_nearest_neighbours=5,
        )

    # Use GraphNetDataModule to load in data
    dm = GraphNeTDataModule(
        dataset_reference=config["dataset_reference"],
        dataset_args={
            "truth": truth,
            "truth_table": "truth",
            "features": features,
            "graph_definition": graph_definition,
            "pulsemaps": [config["pulsemap"]],
            "path": config["path"],
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "shuffle": True,
            "drop_last": True,
        },
        # test_dataloader_kwargs={
        #     "batch_size": config["batch_size"],
        #     "num_workers": config["num_workers"],
        #     "drop_last": True, # skip incomplete final batches
        #     # "collate_fn": collate_fn_debug,
        # },
        validation_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            # "sampler": "SequentialSampler",
            "drop_last": False, # skip incomplete final batches?
        },
    )

    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader

    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    # Define the task.

    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels=["is_hnl"],
        prediction_labels = ["hnl_likelihood"],
        loss_function = BinaryCrossEntropyLoss()
    )

    # Define the full model, which includes the backbone, task(s),
    # along with typical machine learning options such as
    # learning rate optimizers and schedulers.
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
    print("filename is: ")
    print(filename)
    path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/BinaryClassifier/"+ filename + "/"
    os.makedirs(path, exist_ok=True)
    # checkpoint_callback = ModelCheckpoint(dirpath=path)

    # Add a checkpoint at the validation loss minimum
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        dirpath= path,
        filename='best_model-{epoch:02d}-{val_loss:.2f}' # Filename format
    )

    # Training model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        callbacks=[checkpoint_callback],
        **config["fit"],
    )

    # Get predictions
    additional_attributes = [
                                'energy',
                                'position_x',
                                'position_y',
                                'position_z',
                                'azimuth',
                                'zenith',
                                'is_hnl',
                                'pid',
                                'interaction_type',
                                'elasticity',
                                'dbang_decay_length',
                                'event_weight',
                            ]

    assert isinstance(additional_attributes, list)  # mypy
    # Save full model (including weights) to .pth file - not version safe
    # Note: Models saved as .pth files in one version of graphnet
    #       may not be compatible with a different version of graphnet.
    model.save(f"{path}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save_config(f"{path}/model_config.yml")

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config["fit"]["gpus"],
    )

    results.to_csv(f"{path}/results.csv")
    training_predictions = model.predict_as_dataframe(
            training_dataloader,
            # additional_attributes=additional_attributes + ["event_no"],
            gpus=config["fit"]["gpus"],
        )
    # Save predictions and model to file

    # Save results as .csv
    training_predictions.to_csv(f"{path}/training_results.csv")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN classifier without the use of config files.
"""
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default='SRTInIcePulses',
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth", #"mc_truth",#
    )
    parser.add_argument(
        "--filename",
        help="Name of the folder to store results in (default: %(default)s)",
        default="training_results", #"mc_truth",#
    )

    parser.with_standard_arguments(
        ("gpus", torch.cuda.device_count()),
        ("max-epochs", 150),
        ("early-stopping-patience", 3),
        ("batch-size", 8),
    )

    parser.add_argument(
        "--wandb",
        action="store_false",
        help="If True, Weights & Biases are used to track the experiment.",
    )


    args, unknown = parser.parse_known_args()

    main(
        args.pulsemap,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        # args.num_workers,
        args.filename,
        args.wandb,
    )
