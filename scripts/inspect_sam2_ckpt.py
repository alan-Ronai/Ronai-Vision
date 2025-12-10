import torch
import sys

ckpt_path = "models/sam2_small.pt"
try:
    data = torch.load(ckpt_path, map_location="cpu")
except Exception as e:
    print(f"ERROR loading checkpoint: {e}")
    sys.exit(2)

print("Top-level keys:")
if isinstance(data, dict):
    for k in list(data.keys())[:50]:
        print(" -", k)
    # If it contains a 'model' or 'state_dict', print some nested keys
    if "model" in data and isinstance(data["model"], dict):
        print('\nTop keys in data["model"]:')
        for k in list(data["model"].keys())[:50]:
            print(" -", k)
    if "state_dict" in data and isinstance(data["state_dict"], dict):
        print('\nTop keys in data["state_dict"]:')
        for k in list(data["state_dict"].keys())[:50]:
            print(" -", k)
else:
    print(type(data))
