import torch

checkpoint = torch.load("weights/mlp.pth", map_location="cpu")
old_state = checkpoint["model_state_dict"]
new_state = {k.replace("model.", "net."): v for k, v in old_state.items()}
checkpoint["model_state_dict"] = new_state
torch.save(checkpoint, "weights/mlp.pth")
print("완료")
print(list(new_state.keys()))