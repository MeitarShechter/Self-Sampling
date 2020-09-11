import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from modules import DisplacementGenerator, DisplacementGeneratorPP 
from losses import chamfer_distance
from sampler import Sampler

# Need this to create dataset?
# from torch_geometric.data import InMemoryDataset
# from torch_geometric.data import DataLoader


def main():
    ### randomization-related stuff ###
    # random.seed(0)
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)
    # torch.cuda.manual_seed(0)

    ### set all hyperparameters ###
    k = 30
    lr = 0.001
    p_t = 0.8
    p_s = 0.1
    num_epochs = 100
    batch_size = 32
    numInferenceSampling = 20
    num_points = 2000
    opt = 'Adam'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pointCloudPath = "/Users/meitarshechter/Git/Self-Sampling/data/guitar_scale.xyz"

    ### set checkpoint-related stuff ###
    checkpointPath = '/Users/meitarshechter/Git/Self-Sampling/checkpoints/CheckPoint__k_{}__p_s_{}__p_t_{}__bs_{}__num_points_{}__lr_{}__opt_{}__epoch_{}.pt'\
                                                                                .format(k, p_s, p_t, batch_size, num_points, lr, opt, '{}')    
    checkpointLoadPath = '/Users/meitarshechter/Git/Self-Sampling/checkpoints/CheckPoint__k_{}__p_s_{}__p_t_{}__bs_{}__num_points_{}__lr_{}__opt_{}__epoch_{}.pt'\
                                                                                .format(k, p_s, p_t, batch_size, num_points, lr, opt, '20')    
    continue_train = False
    trainedModelPath = '/Users/meitarshechter/Git/Self-Sampling/trained_models/TrainedModel__k_{}__p_s_{}__p_t_{}__bs_{}__num_points_{}__lr_{}__opt_{}.pt'\
                                                                                .format(k, p_s, p_t, batch_size, num_points, lr, opt)

    ### model-realted declarations ###
    model = DisplacementGeneratorPP(num_classes=3).to(device) # expects (batch_size, dim, num_points)
    model.apply(DisplacementGeneratorPP.weights_init) # initialize last FC layer to output near-zero displacements
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[int(elem*num_epochs) for elem in [0.3, 0.6, 0.8]], gamma=0.2)
    crit = chamfer_distance

    ### TensorBoard ###
    summary_path = '/Users/meitarshechter/Git/Self-Sampling/summeries/CheckPoint__k_{}__p_s_{}__p_t_{}__bs_{}__num_points_{}__lr_{}__opt_{}/training'\
                                                                                .format(k, p_s, p_t, batch_size, num_points, lr, opt)
    writer = SummaryWriter(summary_path)

    ### prepare the sampler ###
    sampler = Sampler(pointCloud=pointCloudPath, curvyProbT=p_t, curvyProbS=p_s)
    sampler.preparePointCloud(k=k)

    ### start training ###
    model.train()
    print("Start training")

    if continue_train == True:
        cp = torch.load(checkpointLoadPath, map_location=device)
        trained_epochs = cp['epoch'] + 1
        sd = cp['model_state_dict']
        model.load_state_dict(sd)
        op_sd = cp['optimizer_state_dict']
        optimizer.load_state_dict(op_sd)
        if opt != 'Adam':
            sc_sd = cp['scheduler_state_dict']
            scheduler.load_state_dict(sc_sd)
    else:
        trained_epochs = 0

    for epoch in range(trained_epochs, num_epochs):        
        loss_all = 0.0
        optimizer.zero_grad()

        # sample disjoint source and target sets
        source = sampler.sample(num_points=num_points, batch_size=batch_size, target=False, first_sampling=True)
        target = sampler.sample(num_points=num_points, batch_size=batch_size, target=True, first_sampling=False)
        source = torch.from_numpy(source).to(device) # (batch_size, num_points, dim)
        target = torch.from_numpy(target).to(device) # (batch_size, num_points, dim)

        # PointNet
        # source = torch.from_numpy(source).to(device).unsqueeze(-1)
        # target = torch.from_numpy(target).to(device).unsqueeze(0)

        # calculate predicted target
        displacements = model(source.float()) # (batch_size, num_points, dim)
        predicted_target = source + displacements # (batch_size, num_points, dim)

        # calcualae loss
        loss, _ = crit(target.float(), predicted_target.float(), point_reduction="sum")
        loss.backward()
        loss_all += loss.item()
        print("epoch {}. loss is: {}".format(epoch+1, loss_all))

        optimizer.step()
        
        if opt != "Adam":
            scheduler.step()

        # save checkpoint every 10 epochs
        if epoch % 10 == 9:
            writer.add_scalars('Learning curve', {
            'training loss': loss_all
            }, epoch+1)            

            # save checkpoint
            if opt == "Adam":
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_all,
                }, checkpointPath.format(epoch+1))
            else:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_all,
                'scheduler_state_dict': scheduler.state_dict(),
                }, checkpointPath.format(epoch+1))

    ### save trained model ###
    torch.save(model.state_dict(), trainedModelPath)            

    ### start evaluating ###
    print("Start evaluating")
    # model.load_state_dict(torch.load(trainedModelPath, map_location=device))
    model.eval()

    ### save the upsampled point cloud ###
    newPointCloudFilePath = '/Users/meitarshechter/Git/Self-Sampling/data/upscale_guitar.xyz'
    newPointCloudFile = open(newPointCloudFilePath, 'ab') 

    with torch.no_grad():
        for i in range(numInferenceSampling):
            sample = sampler.sample(num_points=num_points, uniform_sampling=True) 
            sample = torch.from_numpy(sample).to(device) # (batch_size, num_points, dim)

            displacements = model(sample.float())
            predicted_sample = sample + displacements # (batch_size, num_points, dim)
            np.savetxt(newPointCloudFile, predicted_sample.squeeze(0).detach().numpy())

    newPointCloudFile.close()

        # cuda_time = 0.0            
        # cpu_time = 0.0

            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # output = orig_net(images)
            # cuda_time += sum([item.cuda_time for item in prof.function_events])
            # cpu_time += sum([item.cpu_time for item in prof.function_events])

    # writer.add_scalars('Network accuracy', {
    #     'original': o_acc,
    #     'pruned': p_acc
    #     }, 100*p_factor)
    # writer.add_scalars('Network number of parameters', {
    #     'original': o_num_params,
    #     'pruned': p_num_params
    #     }, 100*p_factor)
    # writer.add_scalars('Network GPU time', {
    #     'original': o_cuda_time,
    #     'pruned': p_cuda_time
    #     }, 100*p_factor)
    # writer.add_scalars('Network CPU time', {
    #     'original': o_cpu_time,
    #     'pruned': p_cpu_time
    #     }, 100*p_factor)

    # writer.close()


if __name__ == "__main__":
    main()





