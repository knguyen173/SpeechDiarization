import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda build:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul ok, result mean:", y.mean().item())
else:
    print("No CUDA. You're on CPU.")
