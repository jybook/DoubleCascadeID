from graphnet.data.constants import FEATURES, TRUTH
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