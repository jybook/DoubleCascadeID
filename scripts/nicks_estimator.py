import os
import numpy as np
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from pytorch_lightning.callbacks import ModelCheckpoint

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge,GRIT
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

features = FEATURES.ICECUBE86+["hlc"]
truth = TRUTH.ICECUBE86 + ["is_hnl"]
wandb = True

import glob

if wandb:
    import wandb
    from pytorch_lightning.loggers import WandbLogger




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


sig_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/IceCube/DB_HNL/data/GraphNeT/sqlite_signal/"
bkg_paths = [
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/IceCube/DB_HNL/data/GraphNeT/sqlite_background/22042/",
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/IceCube/DB_HNL/data/GraphNeT/sqlite_background/oscNext/120000/merged/",
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/IceCube/DB_HNL/data/GraphNeT/sqlite_background/oscNext/140000/merged/",
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/IceCube/DB_HNL/data/GraphNeT/sqlite_background/oscNext/160000/merged/",
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/IceCube/DB_HNL/data/GraphNeT/sqlite_background/oscNext/130000/merged/"
]
dataset_paths = glob.glob("%s/*db"%sig_path)
for _path in bkg_paths:
    dataset_paths += glob.glob("%s/*db"%_path)
pulsemap = "InIceDSTPulses"
truth_table = "truth"
target = [
    "ishnl",
         ]
max_epochs = 15
batch_size = 256
gpus = 1
num_workers = 16
early_stopping_patience = 5
# Initialise Weights & Biases (W&B) run
if wandb:
    # Make sure W&B output directory exists
    wandb_dir = "./wandb/"
    os.makedirs(wandb_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="dbang-script",
        entity="nkamp-harvard-university",
        save_dir=wandb_dir,
        log_model=True,
    )


# Configuration
config: Dict[str, Any] = {
    "path": dataset_paths,
    "pulsemap": pulsemap,
    "batch_size": batch_size,
    "num_workers": num_workers,
    "target": target,
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



# Define graph representation
graph_definition = KNNGraph(detector=IceCube86(),
                            input_feature_names=['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area','hlc'],
                            nb_nearest_neighbours=5,
                            distance_as_edge_feature=True
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
        #"selection" : "dbang_decay_length>3",
        # "selection": {'train': 'event_no % 5 > 1',
        # 'test': 'event_no % 5 == 1',
        # 'validation': 'event_no % 5 == 0',
        # },
    },
    train_dataloader_kwargs={
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "shuffle": True,
        "drop_last": True, # skip incomplete final batches
        # "collate_fn": collate_fn_debug,
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
        "drop_last": False, # skip incomplete final batches
        # "collate_fn": collate_fn_debug,
    },
)
training_dataloader = dm.train_dataloader
validation_dataloader = dm.val_dataloader




# Building model

# backbone = DynEdge(
#     nb_inputs=graph_definition.nb_outputs,
#     global_pooling_schemes=["min", "max", "mean", "sum"],
# )
backbone = GRIT(
    nb_inputs=graph_definition.nb_outputs,
    hidden_dim=32,
    ksteps=6,
)

# Define the task.
# Here an energy reconstruction, with a LogCoshLoss function.
# The target and prediction are transformed using the log10 function.
# When infering the prediction is transformed back to the
# original scale using 10^x.
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

path = "./BiggerClassifier/"
os.makedirs(path, exist_ok=True)
checkpoint_callback = ModelCheckpoint(dirpath=path)

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
                            # 'pid',
                            # 'event_time',
                            # 'sim_type',
                            # 'interaction_type',
                            # 'elasticity',
                            # 'RunID',
                            # 'SubrunID',
                            # 'EventID',
                            # 'SubEventID',
                            'dbang_decay_length',
                            'track_length',
                            'stopped_muon',
                            'energy_track',
                            'energy_cascade',
                            'inelasticity',
                            'DeepCoreFilter_13',
                            'CascadeFilter_13',
                            'MuonFilter_13',
                            'OnlineL2Filter_17',
                            'L3_oscNext_bool',
                            'L4_oscNext_bool',
                            'L5_oscNext_bool',
                            'L6_oscNext_bool',
                            'L7_oscNext_bool',
                            # 'is_starting',
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


# Save predictions and model to file

# Save results as .csv
results.to_csv(f"{path}/results.csv")