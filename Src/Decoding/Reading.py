from nlb_tools.nwb_interface import NWBDataset

datapath = "/Users/heyunuo/Desktop/Intracortical Neural Decoding Project/Data/MC_Maze/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

dataset = NWBDataset(datapath)

print(dataset)
print(dataset.data.head())
print(dataset.data.columns)