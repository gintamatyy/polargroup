import torch
import numpy as np
import math

centers = torch.zeros([2, 2])
centers[0, 0] = 4
centers[0, 1] = 2
centers[1, 0] = 5
centers[1, 1] = 8
joints = torch.zeros([2, 2])
joints[0, 0] = 7
joints[0, 1] = 4
joints[1, 0] = 2
joints[1, 1] = 4
offsetmap = torch.zeros([10, 12, 2])  # long-range offset
offsetmap2 = torch.zeros([10, 12, 2])  # relay-point offfset
offsetmap3 = torch.zeros([10, 12, 2])  # guided offset
offsetmap4 = torch.zeros([10, 12, 2])  # guided offset
# weightmap = torch.zeros([2, 10, 10])

for i in range(2):
    ct_x = centers[i, 0]
    ct_y = centers[i, 1]
    x = int((joints[i, 0] + centers[i, 0]) / 2)
    y = int((joints[i, 1] + centers[i, 1]) / 2)
    x2 = joints[i, 0]
    y2 = joints[i, 1]
    radius = 1
    start_x = max(int(ct_x - radius), 0)
    start_y = max(int(ct_y - radius), 0)
    end_x = min(int(ct_x + radius + 1), 10)
    end_y = min(int(ct_y + radius + 1), 10)
    start_x2 = max(int(x - radius), 0)
    start_y2 = max(int(y - radius), 0)
    end_x2 = min(int(x + radius + 1), 10)
    end_y2 = min(int(y + radius + 1), 10)
    for pos_x in range(start_x, end_x):
        for pos_y in range(start_y, end_y):
            offset_x = pos_x - x2
            offset_y = pos_y - y2
            offsetmap[pos_y, pos_x, 0] = offset_x
            offsetmap[pos_y, pos_x, 1] = offset_y
            offset_x = pos_x - x
            offset_y = pos_y - y
            offsetmap3[pos_y, pos_x, 0] = offset_x
            offsetmap3[pos_y, pos_x, 1] = offset_y
            # weightmap[0, pos_x, pos_y] = 1
            # weightmap[1, pos_x, pos_y] = 1
    for pos_x2 in range(start_x2, end_x2):
        for pos_y2 in range(start_y2, end_y2):
            offset_x2 = pos_x2 - x2
            offset_y2 = pos_y2 - y2
            offsetmap2[pos_y2, pos_x2, 0] = offset_x2
            offsetmap2[pos_y2, pos_x2, 1] = offset_y2
print('-------inp---------')
print(offsetmap2)
offsetmap4 = offsetmap - offsetmap3
print('-------grid---------')
# shifts_x = torch.arange(0, 12, step=1)
# shifts_y = torch.arange(0, 10, step=1)
# shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
shift_x, shift_y = torch.meshgrid(torch.arange(10), torch.arange(12))
grid = offsetmap4.clone()
grid[:, :, 0] = shift_y - offsetmap4[:, :, 0]
grid[:, :, 1] = shift_x - offsetmap4[:, :, 1]
grid[:, :, 0] = (2 * grid[:, :, 0] / 11) - 1
grid[:, :, 1] = (2 * grid[:, :, 1] / 9) - 1
# grid = grid[:, :, [1, 0]]  # grid的第一维是(12, 10)中的10， 第二维才是12
# new_h = torch.linspace(-1, 1, 10).view(-1, 1).repeat(1, 10)
# new_w = torch.linspace(-1, 1, 10).repeat(10, 1)
# print(new_h)
# print(new_w)
# grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
# grid = grid.unsqueeze(0)
print(grid)
offsetmap2 = offsetmap2.permute(2, 0, 1).unsqueeze(0)
grid = grid.unsqueeze(0)
# inp = torch.arange(10*10).view(1, 1, 10, 10).float()
output = torch.nn.functional.grid_sample(offsetmap2, grid, mode='bilinear', align_corners=True)
# output = torch.nn.functional.grid_sample(inp, grid, mode='bilinear', align_corners=True)
print('-------outp---------')
output = output.numpy()
output0 = output[0, 0, :, :]
output1 = output[0, 1, :, :]
print(output)
print('-------gt---------')
offsetmap4 = offsetmap4.numpy()
gt5 = offsetmap4[:, :, 0]
gt6 = offsetmap4[:, :, 1]
offsetmap3 = offsetmap3.numpy()
gt0 = offsetmap3[:, :, 0]
gt1 = offsetmap3[:, :, 1]
offsetmap2 = offsetmap2.numpy()
gt3 = offsetmap2[0, 0, :, :]
gt4 = offsetmap2[0, 1, :, :]
print(offsetmap)

