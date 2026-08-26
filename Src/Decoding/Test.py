from pynwb import NWBHDF5IO

path = "/Users/heyunuo/Desktop/Intracortical Neural Decoding Project/Data/MC_Maze/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

io = NWBHDF5IO(path, mode="r")
nwb = io.read()

print(nwb)
print("identifier:", nwb.identifier)
print("session_description:", nwb.session_description)
print("subject:", nwb.subject)

io.close()