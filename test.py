import torch
import numpy as np

A = torch.randn((128,1152,16))
B = A.permute(0,2,1)
# print(A.size())
# print(B.size())
C = B.matmul(A).numpy()
print(C.shape)

# e_vals, e_vecs = np.linalg.eig(C)
# #print('e_vals:',e_vals)
# sorted_indices = np.argsort(e_vals,axis=1)
# list = sorted_indices[:,-1]
# #print(sorted_indices)
# #print('list:',list)
# eval=np.zeros(list.shape)
# for i in range(list.shape[0]):
#     eval[i] = e_vals[i][list[i]]
# print(eval)

def topk(mat,k):
    # mat = torch.from_numpy(mat)
    # mat = mat.permute(1,2,0)
    # mat = mat.cpu().detach().numpy()
    e_vals,e_vecs = np.linalg.eig(mat)
    sorted_indices = np.argsort(e_vals)
    return e_vals[sorted_indices[:-k-1:-1]],e_vecs[:,sorted_indices[:-k-1:-1]]

# a = np.array([[1,2,3],[5,8,7],[1,1,1]])
vals,vecs = topk(C,1)
print(vals.shape)
print(vecs.shape)





