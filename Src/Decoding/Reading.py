from nlb_tools.nwb_interface import NWBDataset

datapath = "/Users/heyunuo/Desktop/Intracortical Neural Decoding Project/Data/MC_Maze/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

dataset = NWBDataset(datapath)

print("===== Dataset loaded =====")
print(dataset)

print("\n===== Continuous data =====")
print(dataset.data.head())
print(dataset.data.shape)

print("\n===== Signal types =====")
print(dataset.data.columns.get_level_values("signal_type").unique())

print("\n===== Spikes =====")
spikes = dataset.data["spikes"]
print(spikes.head())
print("spikes shape:", spikes.shape)

print("\n===== Heldout spikes =====")
heldout_spikes = dataset.data["heldout_spikes"]
print("heldout_spikes shape:", heldout_spikes.shape)

print("\n===== Behavior =====")
hand_pos = dataset.data["hand_pos"]
hand_vel = dataset.data["hand_vel"]
print("hand_pos shape:", hand_pos.shape)
print("hand_vel shape:", hand_vel.shape)
print(hand_pos.head())
print(hand_vel.head())

print("\n===== Trial info =====")
print(dataset.trial_info.head())
print(dataset.trial_info.columns)
print("trial_info shape:", dataset.trial_info.shape)

print("\n===== Try make_trial_data =====")
trial_data = dataset.make_trial_data()
print(trial_data.head())
print(trial_data.columns)
print(trial_data.index.names)
print("trial_data shape:", trial_data.shape)
