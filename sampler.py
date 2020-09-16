import numpy as np
import torch

from pytorch3d.ops.knn import knn_points

'''
Sampler class for point cloud sampling as described in https://arxiv.org/pdf/2008.06471.pdf
'''
class Sampler():
    def __init__(self, pointCloud=None, curvyProbT=0.8, curvyProbS=0.2):        
        if isinstance(pointCloud, str):
            pointCloud = np.loadtxt(pointCloud)
        elif isinstance(pointCloud, np.ndarray): 
            pointCloud = pointCloud
        else:
            print("Must receive a point cloud to sample from, either as path to a file or as a numpy array")
            return

        self.pointCloud = pointCloud
        self.curvyProbT = curvyProbT
        self.curvyProbS = curvyProbS
        self.curvy_indices = None
        self.flat_indices = None
        self.all_indices = None

    def preparePointCloud(self, k=5):
        '''
        Preprocess the point cloud.
        Assigns each point its curvy class based on its :attr:`k` nearest neighbors and save
        each class indices.
        '''
        self.pointCloud = self.__assignCurvyClassToPoints(k)
        self.curvy_indices = np.where(self.pointCloud[:,6] == 1)[0] 
        self.flat_indices = np.where(self.pointCloud[:,6] == 0)[0]
        self.all_indices = np.arange(len(self.pointCloud))

    # TODO: change to enum (classes)
    def __assignCurvyClassToPoints(self, k=5):
        pointCloud = torch.from_numpy(self.pointCloud).unsqueeze(0).float()
        _, idx , _ = knn_points(p1=pointCloud, p2=pointCloud, K=k+1) # get indices of k-nn for each point

        pointCloud = pointCloud.squeeze(0)
        points_norm = pointCloud[:, 3:]
        neighbors_norms = pointCloud[idx[0, :, 1:], 3:] # we only want the nx, ny, nz of each neighbor

        cosSim = torch.nn.CosineSimilarity(eps=1e-5)

        # duplicate and tile up the points norms so we can apply cosine similarity in a vectorized fashion
        # taken from https://discuss.pytorch.org/t/repeat-examples-along-batch-dimension/36217/4
        init_dim = points_norm.size(0)
        repeat_idx = [1] * points_norm.dim()
        repeat_idx[0] = k
        points_norm = points_norm.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(k) + i for i in range(init_dim)]))
        points_norm =  torch.index_select(points_norm, 0, order_index)

        # tile up neighbors norms
        neighbors_norms = neighbors_norms.view(-1, 3)

        # compute the angles between each point and its neighbors and sum
        similarities = cosSim(points_norm, neighbors_norms)
        angles = np.arccos(similarities)
        angles[np.where(np.isnan(angles.numpy()))[0]] = 0 # nan seems to occur for similarities of 1 sometimes
        angles = angles.view(-1, k)
        curvatures = np.sum(angles.numpy(), axis=1)

        # set the threshold and assign class for each point
        threshold = np.mean(curvatures)
        classes = curvatures >= threshold # 1 is curvy, 0 is flat

        pointCloud = np.c_[pointCloud, classes]

        return pointCloud

    # TODO: change to enum (target)
    def sample(self, num_points=1, batch_size=1, target=True, uniform_sampling=False, first_sampling=True):
        '''
        Samples :attr:`num_points` :attr:`batch_size` times.
        if :attr:`target` is True then the points are belong to curvy class based on the self.curvyProbT, otherwise based on self.curvyProbS.
        if :attr:`uniform_sampling` is True then the points are sampled uniformly.
        if :attr:`first_sampling` is True then the sampled points are saved, otherwised the saved points cannot be re-sampled. 
        '''
        assert(self.curvy_indices is not None)
        assert(self.flat_indices is not None)
        assert(self.all_indices is not None)

        if num_points >= len(self.curvy_indices) or num_points >= len(self.flat_indices):
            num_points = int(min(self.curvyProbS, self.curvyProbT) * min(len(self.curvy_indices), len(self.flat_indices)))
            print("Warning: number of point to sample is too high, changing it to {}".format(num_points))

        samples = np.zeros((batch_size, num_points, 3))

        if uniform_sampling:
            for i in range(batch_size):
                samples[i, ...] = self.pointCloud[np.random.choice(self.all_indices, num_points, replace=False), :3] 
            return samples
            # return self.pointCloud[np.random.choice(self.all_indices, num_points, replace=False), :3]

        probs = torch.zeros((num_points,))        
        if target:
            probs[:] = self.curvyProbT 
        else:
            probs[:] = self.curvyProbS 

        samples_class = np.zeros((batch_size, num_points))
        for i in range(batch_size):
            samples_class[i] = torch.bernoulli(probs) # 0 is flat, 1 is curvy
        # samples_class = torch.bernoulli(probs) # 0 is flat, 1 is curvy
        
        num_curvy = np.sum(samples_class, axis=1)
        # num_curvy = torch.sum(samples_class, dim=1)
        num_flat = num_points - num_curvy
        # num_curvy = torch.sum(samples_class)
        # num_flat = num_points - num_curvy

        curvy_samples_idx = {}
        flat_samples_idx = {}
        ## curvy_samples_idx = np.zeros((batch_size, num_curvy))
        ## flat_samples_idx = np.zeros((batch_size, num_flat))
        if first_sampling:
            for i in range(batch_size):
                curvy_samples_idx[i] = np.random.choice(self.curvy_indices, int(num_curvy[i]), replace=False)
                flat_samples_idx[i] = np.random.choice(self.flat_indices, int(num_flat[i]), replace=False)
            # curvy_samples_idx = np.random.choice(self.curvy_indices, int(num_curvy), replace=False)
            # flat_samples_idx = np.random.choice(self.flat_indices, int(num_flat), replace=False)
            
            self.curvy_idx_sampled = curvy_samples_idx
            self.flat_idx_sampled = flat_samples_idx
        else:
            for i in range(batch_size):
                curvy_samples_idx[i] = np.random.choice(np.array(list(set(self.curvy_indices) - set(self.curvy_idx_sampled[i]))), int(num_curvy[i]), replace=False)
                flat_samples_idx[i] = np.random.choice(np.array(list(set(self.flat_indices) - set(self.flat_idx_sampled[i]))), int(num_flat[i]), replace=False)
            # curvy_samples_idx = np.random.choice(np.array(list(set(self.curvy_indices) - set(self.curvy_idx_sampled))), int(num_curvy), replace=False)
            # flat_samples_idx = np.random.choice(np.array(list(set(self.flat_indices) - set(self.flat_idx_sampled))), int(num_flat), replace=False)

        samples_idx = {}
        for i in range(batch_size):
            samples_idx[i] = np.concatenate((curvy_samples_idx[i], flat_samples_idx[i]))
        # samples_idx = np.concatenate((curvy_samples_idx, flat_samples_idx))

        for i in range(batch_size):
            samples[i, ...] = self.pointCloud[samples_idx[i], :3]
        # samples = self.pointCloud[samples_idx, :3]

        return samples
