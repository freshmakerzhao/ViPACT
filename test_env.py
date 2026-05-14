import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())

m = torch.nn.MultiheadAttention(512, 8).cuda()
x = torch.randn(102, 4, 512, device="cuda")
mask = torch.zeros(4, 102, dtype=torch.bool, device="cuda")
y = m(x, x, x, key_padding_mask=mask)
torch.cuda.synchronize()
print(y[0].shape)