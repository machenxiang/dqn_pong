import gymnasium as gym  # 修改这里
from gymnasium import spaces  # 修改这里
import numpy as np

import torch
import torch.nn.functional as F


# a= torch.zeros(2,3)
# print(a)
# b=np.array([[1,2],[3,4]])
# tensor_from_numpy = torch.from_numpy(b)
# print(tensor_from_numpy)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# a=torch.ones(2,3)
# b=torch.ones(2,3)

# print(a+b)
# print(a*b)
# print(a.t())

# tensor_requires_grad = torch.tensor([1.0],requires_grad=True)
# tensor_result = tensor_requires_grad*3
# tensor_result.backward()
# print(tensor_requires_grad.grad)

# x =torch.ones(2,2,requires_grad=True).detach()
# y = x+2

# z= y*y*3
# out=z.mean()
# print(out)

# out.backward()
# print(x.grad)
# # x.grad.zero_()
# # out2 = (x*x).sum()
# # out2.backward()
# # print(x.grad)
# a= torch.tensor([[1,2],[3,4]])
# print(a)
# b= torch.rand(2,2)
# print(b)
# c= torch.arange(0,10,0.5)
# print(c)
# d= torch.linspace(0,10,20)
# print(d)
# ee=np.array([[1,2],[3,4]])
# e= torch.from_numpy(ee)
# print(e)

# tensor2d = torch.ones(2,2)
# print(tensor2d)
# print("shape:",tensor2d.shape)

# tensor3d = torch.stack([tensor2d,tensor2d+1])
# print(tensor3d)
# print("shape:",tensor3d.shape)


# tensor4d = torch.stack([tensor3d,tensor3d+3])
# print(tensor4d)
# print("shape:",tensor4d.shape)
device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
# a= torch.ones(2,2,device =device)
# b = torch.ones(2,2,device =device)
# c= a.matmul(b)
# d= torch.dot(a,b)

# print(c)
# print(d)

a= torch.tensor([[1,2,6],[3,4,1]])
print(a)

b= a.view(6,-1)
print(b)


c= a.reshape(3,2)
print(c)

d = a.unsqueeze(1)
print(d)



