from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.data.dataset import EnsembleDataset
# from graphnet.models.detector.prometheus  import  Prometheus
from graphnet.models.detector.icecube  import  IceCube86
from graphnet.models.graphs  import  KNNGraph
from graphnet.models.graphs.nodes  import  NodesAsPulses

import glob

MC_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/HNL/parquet/'
SET_DIRS = ['190301', '190302', '190303', '190304', '190305', '190306', '190307', '190308', 'oscNext']

# Combine all truth-level MC sets
truth_files = []
for dir in SET_DIRS:
    truth_files.extend(glob.glob(MC_DIR + dir + '/truth/*'))

# This is just the keys of the parquet file - not sure this is the right thing, but there are a lot of fields in the other dataset definitions that we dont' appear to have, which is concerning. Like maybe it didn't read the MCTree correctly? At least it's consistent accross the HNL and oscnext sets
truth_variables = ['energy', 'position_x', 'position_y', 'position_z', 'azimuth', 'zenith',
       'pid', 'event_time', 'sim_type', 'interaction_type', 'elasticity',
       'RunID', 'SubrunID', 'EventID', 'SubEventID', 'dbang_decay_length',
       'track_length', 'stopped_muon', 'energy_track', 'energy_cascade',
       'inelasticity', 'DeepCoreFilter_13', 'CascadeFilter_13',
       'MuonFilter_13', 'OnlineL2Filter_17', 'L3_oscNext_bool',
       'L4_oscNext_bool', 'L5_oscNext_bool', 'L6_oscNext_bool',
       'L7_oscNext_bool']

graph_definition = KNNGraph(
    detector=IceCube86(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
)

datasets = []
for file in truth_files:
    print(file)
    dataset = ParquetDataset(
        path=file,
        pulsemaps="SRTInIcePulses",
        truth_table="mc_truth",
        features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"],
        truth=truth_variables,
        graph_definition = graph_definition,
    )
    datasets.append(dataset)
    break
datasets[0].config.dump("sample_HNL.yaml")
ensemble_dataset = EnsembleDataset(datasets=datasets)
# ensemble_dataset.config.dump('merged_HNL_oscNext.yml')