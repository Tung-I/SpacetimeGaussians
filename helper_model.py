# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch 
from thirdparty.gaussian_splatting.utils.graphics_utils import BasicPointCloud
import numpy as np
from simple_knn._C import distCUDA2
from mmcv.ops import knn
import torch.nn as nn



class Sandwich(nn.Module):
    """ 3D point feature decoder
    """
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich, self).__init__()
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = self.sigmoid(result) 
        return result


class Sandwichnoact(nn.Module):
    """Clamping verison without sigmoid
    """
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoact, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()



    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = torch.clamp(result, min=0.0, max=1.0)
        return result

class Sandwichnoactss(nn.Module):
    """ Without clamping and sigmoid
    """
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoactss, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)  
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        result = albedo + specular
        return result
    
    
class RGBDecoderVRayShift(nn.Module):
    """Another rgb model lower than sandwich,
    """
    def __init__(self, dim, outdim=3, bias=False):
        super(RGBDecoderVRayShift, self).__init__()
        
        self.mlp1 = nn.Conv2d(dim, outdim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(15, outdim, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

        self.dwconv1 = nn.Conv2d(9, 9, kernel_size=1, bias=bias)

    def forward(self, input, rays, t=None):
        x = self.dwconv1(input) + input 
        albeado = self.mlp1(x)
        specualr = torch.cat([x, rays], dim=1)
        specualr = self.mlp2(specualr)

        finalfeature = torch.cat([albeado, specualr], dim=1)
        result = self.mlp3(finalfeature)
        result = self.sigmoid(result)   
        return result 
    

def interpolate_point(pcd, N=4):
    """Interpolate the point cloud over time and space
    Args:
        pcd: BasicPointCloud
        N: int, the number of frames to interpolate
    Returns:
        newpcd: BasicPointCloud
    """
    # Extracts the point positions (xyz), colors, normals, and times from the point cloud `pcd`
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    timestamps = np.unique(oldtime)
    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    print(f'pcd shape: {oldxyz.shape}')
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time  # Create a mask to select all points corresponding to the current timestamp
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx == 0:  # If this is the first timestamp, simply add the points, colors, normals, and times to the new lists
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])
        else:   # For other timestamps, we interpolate by selecting points based on nearest neighbors
            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3

            # Perform a k-nearest neighbors (KNN) search to find the 2 nearest neighbors for each point
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            # Extract the indices of the nearest neighbors
            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1  

            # Calculate the Euclidean distance between each point and its nearest neighbor
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            # get the sorted distances
            diff_sorted, _ = torch.sort(spatialdistance) 
            N = spatialdistance.shape[0]
            num_take = int(N * 0.25)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])
            newtime.append(oldtime[selectedmask][masksnumpy])
            
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)
    print("Interpolated point cloud shape: ", newpcd.points.shape)

    return newpcd



def interpolate_pointv3(pcd, N=4, m=0.25):
    
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx % N == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

        else:
            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            M = spatialdistance.shape[0]
            num_take = int(M * m)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])
            newtime.append(oldtime[selectedmask][masksnumpy])
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd




def interpolate_partuse(pcd, N=4):
    # used in ablation study
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)

    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx % N == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

        else:
            pass
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  

    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd



def padding_point(pcd, N=4):
    
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)
    totallength = len(timestamps)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        

        if timeidx != 0 and timeidx != len(timestamps) - 1:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

             
        else:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3


            xyznnpoints = knn(2, xyzinput, xyzinput, False)


            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            N = spatialdistance.shape[0]
            num_take = int(N * 0.125)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])

            if timeidx == 0:
                newtime.append(oldtime[selectedmask][masksnumpy] - (1 / totallength)) 
            else :
                newtime.append(oldtime[selectedmask][masksnumpy] + (1 / totallength))
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd

def getcolormodel(rgbfuntion):
    if rgbfuntion == "sandwich":
        rgbdecoder = Sandwich(9,3)
    elif rgbfuntion == "sandwichnoact":
        rgbdecoder = Sandwichnoact(9,3)
    elif rgbfuntion == "sandwichnoactss":
        rgbdecoder = Sandwichnoactss(9,3)
    else :
        return None 
    return rgbdecoder


# Converting between pixel coordinates (pix) and normalized device coordinates (ndc)
def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5



