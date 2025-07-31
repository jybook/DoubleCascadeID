'''
Script to remove events with the invisible three-neutrino decay mode from the training set
Also applies a few quality cuts
Can read in files at any level, as long as it includes the 'EventProperties' in the frame (set --keep-dc and --keep-lowup flags if below L1).
Requires LI-HNL to read decay mode from 'EventProperties' 
'''

#!/bin/sh /n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/setup.sh
#METAPROJECT /data/user/jbook/i3/build

import time
import os, glob
# import numpy as np

from icecube.icetray import I3Tray, I3Frame, I3PacketModule
from icecube import icetray, dataclasses, LeptonInjector, dataio

from icecube.icetray import I3Units
# from icecube.LeptonInjector import weight_hnl_generation, load_generation_weighting_files, weight_hnl_lifetime_framewise
# from icecube.LeptonInjector.oscnext_frame_objects_weighting import create_weight_dict
# from icecube.hdfwriter import I3SimHDFWriter

from optparse import OptionParser

# Run filters on Q frames, dropping P's when an event is cut.
class DAQFilters(icetray.I3PacketModule):
    def __init__(self, context):
       I3PacketModule.__init__(self, context, I3Frame.DAQ)
       self.AddParameter("cut_energy", "energy cut value", "")
       self.AddOutBox("OutBox")
       self.Configure()

    def Configure(self):
        self.cut_energy = self.GetParameter("cut_energy")

    def _skip_nununu_frames(self, frame):
        '''
        Skip frames with no chance of double cascade.
        Inputs:
            The frame
        Returns:
            False if the decay mode is 1 (corresponding to the three neutrino mode), and true otherwise
        '''
        if not frame.Has("EventProperties"):
            # print("ack! not a daq frame")
            return False
        event_properties = frame["EventProperties"]
        decay_channel = event_properties.decay_channel
        
        return (decay_channel != 1)
    
    # function to skip frames with weight = 0
    def _drop_weight_zero_events(self, frame):
        '''
        Skip frames with LifetimeWeight = 0 (this happens if a nonphysical event isn't flagged properly)
        '''
        if not frame.Has("I3MCWeightDict"):
            print("weight dict not found!")
            return False
        
        weight_dict = frame["I3MCWeightDict"]
        if weight_dict["LifetimeWeight_1e-03"] == 0.0:
            return False
        else:
            return True

    def _energy_cut(self, frame):
        '''
        Skip frames whith low energies
        Inputs:
            The frame, the energy to cut (lower limit)
        Returns:
            event energy > energy
        '''

        if not frame.Has("EventProperties"):
            # print("ack! not a daq frame")
            return False
        event_properties = frame["EventProperties"]       
        return (event_properties.totalEnergy > self.cut_energy)
    
    def FramePacket(self, frames):

        for fr in frames:
            if fr.Stop == icetray.I3Frame.DAQ:
                nununu_flag = self._skip_nununu_frames(fr)
                energy_flag = self._energy_cut(fr)
                weight_flag = self._drop_weight_zero_events(fr)
            fr.Put("use_in_training", icetray.I3Bool(all([nununu_flag, energy_flag, weight_flag])))
            self.PushFrame(fr)

        # return(all([nununu_flag, energy_flag]))
        return False

# Cuts on Physics Frames
def deepcore_cut(frame):
        '''
        Skip frames which don't trigger the deepcore filter
        Requires frame at L1 or above
        Inputs:
            The frame
        Returns:
            True if DC filter passes, false otherwise
        '''
        if not frame.Has("FilterMask"):
            return False
        filter_mask = frame["FilterMask"]

        return filter_mask["DeepCoreFilter_13"].condition_passed


def lowup_cut(frame):
        '''
        Skip frames which don't trigger the lowup filter
        Requires frame at L1 or above
        Inputs:
            The frame
        Returns:
            True if LowUp filter passes, false otherwise
        '''
        if not frame.Has("FilterMask"):
            return False
        filter_mask = frame["FilterMask"]

        return filter_mask["LowUp_13"].condition_passed

def drop_frames(frame):
        '''
        Drop which failed cuts on the DAQ frame, above
        '''
        if not frame.Has("use_in_training"):
            return False
        return frame["use_in_training"]==True #this forces it to send a boolean C++ can recognize instead of an I3Bool

def main():
    '''
    Main function: Run OptionParser, load the files, skip empty frames, leaving everything else unchanged.
    '''
    
    from optparse import OptionParser

    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("-i", "--infile", dest="infile", type="string", help="Infile name or folder. Should be L1 or above.")
    parser.add_option("-o", "--outfile", dest="outfile", type="string", help="Outfile name, requires .i3.zst file extension")
    parser.add_option("-e", "--energy", dest="energy", default="10.0", type="string", help="Lower energy cut, in GeV. [default: %default]")
    parser.add_option("--keep-dc", action="store_false", dest="dc_filter", default=True, help="Keep events NOT passing deepcore filter")
    parser.add_option("--keep-lowup", action="store_false", dest="lowup_filter", default=True, help="Keep events NOT passing lowup filter")

    options,args = parser.parse_args()

    infile 		= options.infile
    outfile 	= options.outfile
    dc_filter   = options.dc_filter
    lowup_filter= options.lowup_filter
    energy      = float(options.energy)

    if not os.path.isfile(infile):
        if os.path.isdir(infile):
            # Get all i3 files in the specified folder
            filename_list = glob.glob(infile+"*.i3*")
        else:
            raise Exception("Please enter an existing folder or filename")
    else:
        filename_list = [infile]

    t0 = time.perf_counter() # time.clock()

    print('Starting ...')
    print('Infile(s): {}'.format(infile))
    print('Outfile: {}'.format(outfile))

    # now start the actual tray processing
    tray = I3Tray()
    tray.Add(dataio.I3Reader, "reader", filenamelist=filename_list)

    # run quality cuts
    tray.AddModule(
        DAQFilters,
        'daq_filters',
        cut_energy=energy,
    )
    tray.AddModule(
        drop_frames,
        'drop_frames'
    )
    if lowup_filter:
        tray.AddModule(
            lowup_cut
        )
    
    if dc_filter:
        tray.AddModule(
            deepcore_cut
        )

    tray.Add(
        "Delete", "cleanup", keys=['use_in_training']
        )
    
    # Write the physics and DAQ frames
    tray.AddModule(
        "I3Writer",
        "EventWriter",
        filename=outfile,
        Streams=[
            icetray.I3Frame.DAQ,
            icetray.I3Frame.Physics,
            icetray.I3Frame.TrayInfo,
            icetray.I3Frame.Simulation,
        ],
        DropOrphanStreams=[icetray.I3Frame.DAQ],
    )

    tray.Execute()
    tray.Finish()


    t1 = time.perf_counter() # time.clock()

    print("Time it took: {}s".format(t1-t0))
    print('done ...')


if __name__ == '__main__':

    main()
