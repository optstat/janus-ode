import torch
class VecOpDual():
    def __init__(self, r, sidx):
        super().__init__()
        self.r = r                  #The dimensionality of the is M x D
        #Store the dual part in a sparse tensor
        indices = []
        values  = []
        ilst = []
        nlst = []
        for i in range(r.shape[0]):
            ilst.append(i)
            nlst.append(sidx+i)
            values.append(1.0)

        indices.append(ilst)
        indices.append(nlst)
        self.values = values
        self.indices = indices
        nsz = r.shape+(r.shape[0],)

        self.d = torch.sparse_coo_tensor(indices, values, size=nsz, device=r.device)

    def resize(self, N):
        nsz = self.r.shape + (N,)
        #This is a padding exercise and indices do not change
        self.d = torch.sparse_coo_tensor(self.indices, self.values, size=nsz, device=self.r.device)

    def __add__(self, other):
        if isinstance(other, TensorDual):
            r = other.r+self.r
            d = other.d
            for _ in self.values: #Loop over the non zero values.  This is fast
                 i = self.indices[0][idx]
                 j = self.indices[1][idx]
                 n = self.indices[2][idx]
                 d[:,i,n] +=1.0
                 idx = idx+1
            return TensorDual(r, d)
        else:
            raise InvalidOperationError('VecOpDual only support multiplication with ParameterDual')
