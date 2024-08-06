import mne


filePath = input("Enter EDF file: ")
filePath = filePath.replace("\\","/")
raw = mne.io.read_raw_edf(filePath, preload=True)

raw.plot()
input("Press enter to exit.")