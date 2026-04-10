import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# centers = torch.zeros([2, 2])
# centers[0, 0] = 4
# centers[0, 1] = 2
# centers[1, 0] = 5
# centers[1, 1] = 8
joints = torch.zeros([2, 2])
joints[0, 0] = 91
joints[0, 1] = 45
joints[1, 0] = 196
joints[1, 1] = 151
offsetmap = torch.zeros([325, 389, 2])  # long-range offset
# offsetmap2 = torch.zeros([10, 12, 2])  # relay-point offfset
# offsetmap3 = torch.zeros([10, 12, 2])  # guided offset
# offsetmap4 = torch.zeros([10, 12, 2])  # guided offset
# # weightmap = torch.zeros([2, 10, 10])


ct_x = joints[0, 0]
ct_y = joints[0, 1]
jt_x = joints[1, 0]
jt_y = joints[1, 1]

radius = 4
start_x = max(int(ct_x - radius), 0)
start_y = max(int(ct_y - radius), 0)
end_x = min(int(ct_x + radius + 1), 325)
end_y = min(int(ct_y + radius + 1), 389)

for pos_x in range(start_x, end_x):
    for pos_y in range(start_y, end_y):
        offset_x = pos_x - jt_x
        offset_y = pos_y - jt_y
        offsetmap[pos_y, pos_x, 0] = offset_x
        offsetmap[pos_y, pos_x, 1] = offset_y

print('-------inp---------')
print(offsetmap)

print('-------grid---------')
# shifts_x = torch.arange(0, 12, step=1)
# shifts_y = torch.arange(0, 10, step=1)
# shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
shift_x, shift_y = torch.meshgrid(torch.arange(325), torch.arange(389))
grid = offsetmap.clone()
grid[:, :, 0] = shift_y - offsetmap[:, :, 0]
grid[:, :, 1] = shift_x - offsetmap[:, :, 1]
grid[:, :, 0] = (2 * grid[:, :, 0] / 389) - 1
grid[:, :, 1] = (2 * grid[:, :, 1] / 325) - 1
# grid = grid[:, :, [1, 0]]  # grid的第一维是(12, 10)中的10， 第二维才是12
# new_h = torch.linspace(-1, 1, 10).view(-1, 1).repeat(1, 10)
# new_w = torch.linspace(-1, 1, 10).repeat(10, 1)
# print(new_h)
# print(new_w)
# grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
# grid = grid.unsqueeze(0)
print(grid)
offsetmap2 = offsetmap.permute(2, 0, 1).unsqueeze(0)
grid = grid.unsqueeze(0)
# inp = torch.arange(10*10).view(1, 1, 10, 10).float()
output = torch.nn.functional.grid_sample(offsetmap2, grid, mode='bilinear', align_corners=True)
# output = torch.nn.functional.grid_sample(inp, grid, mode='bilinear', align_corners=True)
print('-------outp---------')
grid_x_np = grid[0,:,:,0].numpy()
grid_y_np = grid[0,:,:,1].numpy()
plt.imshow(grid_x_np)
plt.show()
plt.imshow(grid_y_np)
plt.show()
# print(output)
# print('-------gt---------')
# offsetmap4 = offsetmap4.numpy()
# gt5 = offsetmap4[:, :, 0]
# gt6 = offsetmap4[:, :, 1]
# offsetmap3 = offsetmap3.numpy()
# gt0 = offsetmap3[:, :, 0]
# gt1 = offsetmap3[:, :, 1]
# offsetmap2 = offsetmap2.numpy()
# gt3 = offsetmap2[0, 0, :, :]
# gt4 = offsetmap2[0, 1, :, :]
print(grid)

