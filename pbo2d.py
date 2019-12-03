# Binary Classification using Gaussian Processes in GPyTorch
# Dataset is a simulated 2D sinusoid 
# Classes are learned by active learning. Points in 2D space are sampled
# by finding point of the maximum variance of the GP classifier.
# Uses L-BFGS to find maximum points in 2D space

from gpytorch.mlls.variational_elbo import VariationalELBO
import math
import torch
import torch.distributions as dist
import gpytorch
from matplotlib import pyplot as plt
import os
from scipy.optimize import minimize

import pandas as pd
import numpy as np
import seaborn as sns
import sys
import time
import csv
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

torch.manual_seed(4);
def create_hypercube_grid(bounds, num_points, n_dims):
    axpoints = torch.linspace(bounds[0], bounds[1], num_points);
    x = torch.zeros(num_points ** n_dims, n_dims);
    #print(axpoints.unsqueeze(1).repeat(num_points ** (n_dims - 0 - 1), num_points ** (0 + 1)).view(-1));
    for i in range(n_dims):
        x[:,i].copy_(axpoints.unsqueeze(1).repeat(num_points ** (n_dims - 1 - i), num_points ** i).view(-1))

    return x;


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x, lengthscale=None):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if lengthscale is not None:
        	self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale=lengthscale))
        else:
        	self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def set_data(self, new_X):
        self.__init__(new_X);

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class VSGPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x, lengthscale=None):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=True)
        super(VSGPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if lengthscale is not None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale=lengthscale))
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class GPClassifier:
    def __init__(self, sparse=False, n_dims=1):
        # Initialize model and likelihood
        
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.likelihood.train();
        self.sparse = sparse;
        self.n_dims = n_dims
        self.X = None;
        self.y = None;

        self.training_time = 0.0;

    def fit(self, X, y, lengthscale=None):
        
        if self.sparse:
            perm = torch.randperm(X.size(0))
            idx = perm[:50]
            self.inducing_points = X[idx];
            self.inducing_points.sort();
            class_model = VSGPClassificationModel;
        else:
            self.inducing_points = X;
            class_model = GPClassificationModel;

        self.X = X;
        self.y = y;

        self.model = class_model(self.inducing_points, lengthscale);
        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the amount of training data
        self.mll = VariationalELBO(self.likelihood, self.model, self.y.numel())

        self.train();

    def train(self, num_steps=100):

        start = time.time();
        for i in range(num_steps):
            # Zero backpropped gradients from previous iteration
            self.optimizer.zero_grad()
            # Get predictive output
            output = self.model(self.X)
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.y)
            loss.backward()
            #print('Iter %d/%d - Loss: %.3f' % (i + 1, num_steps, loss.item()))
            self.optimizer.step()

        end = time.time()
        self.training_time += end - start;

    def update_post(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float();

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float();

        if self.X is not None:
            #print(X.shape, self.X.shape)
            assert X.shape[1] == self.X.shape[1], 'Input shape does not match'

        else:
            self.fit(X, y);
            return 0;

        lengthscale = self.model.covar_module.base_kernel.lengthscale.item()
        self.fit(torch.cat((self.X, X), dim=0), torch.cat((self.y, y), dim=0), lengthscale=lengthscale);

    def forward(self, X):
        with torch.no_grad():
            pred_f = self.model(X);

            return pred_f;

    def predict(self, X, y=None, acc=True):

        if acc:
            assert y is not None, 'target labels required to calculate accuracy'

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            # Test x are regularly spaced by 0.01 0,1 inclusive
            test_x = X
            # Get classification predictions
            pred_f = self.model(test_x);
            
            observed_pred = self.likelihood(pred_f)

            # Get the predicted labels (probabilites of belonging to the positive class)
            # Transform these probabilities to be 0/1 labels
            pred_labels = observed_pred.mean.ge(0.5).float()

            if acc:
                acc_pctg = (pred_labels == y).sum() * 100 / len(y);
                return pred_labels, acc_pctg
            else:
                return pred_labels

def levy(x):
    if not isinstance(x, torch.Tensor): x = torch.tensor(x).float();    
    if len(x.shape) < 2: x = x.unsqueeze(0);
    assert x.shape[1] == 2, 'Input should be of dimension 2'

    w = lambda x: 1. + (x - 1.)/4.; 
    result = torch.sin(math.pi * w(x[:,0])) ** 2;
    result += torch.sum(((w(x[:,:-1])-1)**2) * (1. + 10*torch.sin(math.pi * w(x[:,:-1]) + 1.) ** 2), dim=1);
    result += ((w(x[:,-1])-1)**2) * (1. + torch.sin(2. * math.pi * w(x[:,-1])) ** 2)

    return result;

def dixon_price(x):
    if not isinstance(x, torch.Tensor): x = torch.tensor(x).float();    
    if len(x.shape) < 2: x = x.unsqueeze(0);
    assert x.shape[1] == 2, 'Input should be of dimension 2'

    result = (x[:,0] - 1.) ** 2;
    result += 2. * (2. * x[:,1] ** 2 - x[:,0]) ** 2;
    return result;


def get_preference_prob(obj_f, x1, x2, minimum=True):
   
    diff = obj_f(x1) - obj_f(x2);
    return diff, torch.sigmoid((int(not minimum)*2-1)*diff);

if __name__ == '__main__':
    cmap = plt.get_cmap('jet');
    #obj_f = lambda x: (6*x - 2)**2 * torch.sin(12*x - 4)
    obj_f = dixon_price
    bounds = [(-10.0,10.0),(-10.0,10.0)];
    grid_size = 100;
    gridpoints = create_hypercube_grid(bounds[0], 100,2);
    grid_size = grid_size ** 2;
    x = gridpoints[torch.linspace(0, grid_size-1, 810).long()];

    def random_duel(bounds, *args):
        sample1 = torch.distributions.uniform.Uniform(torch.tensor([bounds[0][0]]), torch.tensor([bounds[0][1]])).sample(sample_shape=(2,));
        sample2 = torch.distributions.uniform.Uniform(torch.tensor([bounds[1][0]]), torch.tensor([bounds[1][1]])).sample(sample_shape=(2,));
        #sample2 = torch.distributions.uniform.Uniform(torch.tensor([bounds[0]]), torch.tensor([bounds[1]])).sample()
        sample = torch.cat((sample1, sample2), dim=1)

        return sample

    def duelling_thompson(bounds, gpc, fig, cmap, selected, prefs):
        n = 100
        grid = torch.zeros(n ** 2, 2)
        axpoints = torch.linspace(bounds[0], bounds[1], n)
        grid[:, 0].copy_(axpoints.repeat(n))
        grid[:, 1].copy_(axpoints.unsqueeze(1).repeat(1, n).view(-1))
    
        post_f = gpc.forward(grid);
        mean = post_f.mean;
        var = post_f.variance;
        sample = [dist.normal.Normal(m, v).sample().tolist() for m, v in zip(mean,var)];
        sample = torch.tensor(sample).reshape(n,n);
        #sample = post_f.mean.reshape(n,n);
        #sample = post_f.sample().reshape(n,n);
        probs = torch.sigmoid(sample);
        #var = post_f.covariance_matrix.diag().reshape(n,n);
        var = var.reshape(n,n);

        copeland_scores = probs.sum(dim=0);
        x1_idx = torch.argmax(copeland_scores);
        x1 = axpoints[x1_idx].unsqueeze(0);
        x2_idx = torch.argmax(var[:, x1_idx]);
        #x2_idx = torch.argmax(var[x1_idx]);
        x2 = axpoints[x2_idx].unsqueeze(0);
        """
        plt.figure(1);
        plt.pause(1);
        plt.clf()
        plt.subplot(1,3,1);
        pcm = plt.pcolormesh(axpoints.numpy(), axpoints.numpy(), probs.numpy(), cmap=cmap);
        #plt.scatter(x1,0, color='white');
        plt.axvline(x1.numpy(), color='white', linestyle='--');

        plt.subplot(1,3,2);
        plt.plot(axpoints.numpy(), copeland_scores.numpy());
        plt.axvline(x1.numpy(), color='black', linestyle='--')

        plt.subplot(1,3,3);
        plt.pcolormesh(axpoints.numpy(), axpoints.numpy(), var.numpy(), cmap=cmap);
        plt.axvline(x1.numpy(), color='white', linestyle='--');
        plt.scatter(x1.numpy(), x2.numpy());
        selected = torch.tensor(selected).squeeze(1);
        prefs = torch.tensor(prefs);
        plt.scatter(selected[:,0], selected[:,1], s=3, color='black', marker='^');

        plt.show(block=False);
        """
        
        return torch.cat((x1, x2), dim=0).reshape(1,-1);

    def copeland(gpc, x, bounds, num_anchors=500, soft=True, n_dims=1):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor([x]).float();

        y = torch.zeros(x.shape[0]);
        #anchors = torch.linspace(bounds[0], bounds[1], num_anchors).reshape(-1,1);
        anchor_idx = torch.linspace(0, grid_size-1, num_anchors).long();
        anchors = gridpoints[anchor_idx].reshape(-1,n_dims);
        #print(x.shape);
        for i, x_ in enumerate(x):
            
            inp = x_.repeat(num_anchors, 1);
            #print(inp.shape, anchors.shape);
            test_x = torch.cat((inp, anchors), dim=1);
            
            post_f = gpc.forward(test_x);
            probs_f = gpc.likelihood(post_f);
            
            probs = probs_f.mean;
            #probs = torch.sigmoid(post_f.mean);
            
            preds = probs.ge(0.5).float();
            
            try:
                y[i] = (1.0/float(num_anchors)) * probs.sum() if soft else preds.sum()/len(preds);
            except IndexError:
                y = (1.0/float(num_anchors)) * probs.sum() if soft else preds.sum()/len(preds);

        return y;

    def condorcet(gpc, bounds, n_dims):

        def func(X):
            return -1.0 * copeland(gpc, X, bounds, n_dims=n_dims);

        num_repeats = 50;
        max_cop = float("inf");
        max_x = float("inf");

        guesses_idx = torch.distributions.uniform.Uniform(torch.tensor([0.]), torch.tensor([grid_size-1.])).sample(torch.Size([num_repeats,1])).long();
        init_guesses = gridpoints[guesses_idx];
        for x0 in init_guesses:
            res = minimize(func, x0.numpy(), bounds=bounds);
            y = func(res.x)
            if y < max_cop:
                max_cop = y;
                max_x = res.x;

        return max_x


    num_duels = 50;
    num_trials = 1;
    fig = plt.figure(1, figsize=(12,4));
    #fig = plt.figure(1, figsize=(8,6));
    w_rand = [];
    w_dts = [];
    times = [[],[]]

    #noise_probs = 0.2;

    n_dims=2;
    for k in range(num_trials):
        iter_times = [[],[]];
        start = time.time();
        winners = [np.zeros((num_duels, n_dims)), np.zeros((num_duels, n_dims))];
        cp_score = [0,0];
        names=['Random', 'DTS'];
        gpc = [GPClassifier(sparse=False, n_dims=4)]#, GPClassifier(sparse=False, n_dims=4)];
        ac_funcs = [random_duel, duelling_thompson];
        selected = [[],[]]
        prefs_s = [[],[]]
        duel = random_duel(bounds);
        error_count = 0;

        for i in range(num_duels):
            #print('---------- Iteration {} ---------'.format(i+1));

            for j in range(len(gpc)):
                iter_start = time.time();
                
                if i != 0:
                    #duel = ac_funcs[j](bounds, gpc[j],None,cmap, selected[j], prefs_s[j])
                    duel = random_duel(bounds); 

                #print('{}:'.format(names[j]))
                selected[j].append(duel.squeeze(0).tolist());
                diff, prob = get_preference_prob(obj_f, duel[0],duel[1]);
                preference = prob.ge(0.5).float();
                #flip_pref = dist.Bernoulli(noise_probs).sample();
                #preference = (1. - flip_pref) * preference + flip_pref * (1. - preference);
                #preference = torch.distributions.Bernoulli(prob).sample().float()
                error_count += float((preference != prob.ge(0.5).float()).squeeze(0));
                prefs_s[j].append(preference.squeeze(0).tolist());

                duel_2 = torch.flip(duel, dims=(0,));
                diff_2, prob_2 = get_preference_prob(obj_f, duel_2[0],duel_2[1]);
                #preference_2 = prob_2.ge(0.5).float();

                preference_2 = 1. - preference;
                selected[j].append(duel_2.squeeze(0).tolist());
                prefs_s[j].append(preference_2.squeeze(0).tolist());

                duels = torch.cat((duel.reshape(1,-1), duel_2.reshape(1,-1)), dim=0);
                preferences = torch.cat((preference, preference_2), dim=0)
                """
                print('Selected Duel: {}'.format(duel.tolist()))
                print('Diff: {}, Prob: {}, Preference: {}'.format(diff.tolist(), prob.tolist(), preference.tolist()))

                print('Flipped Duel: {}'.format(duel_2.tolist()))
                print('Diff: {}, Prob: {}, Preference: {}\n'.format(diff_2.tolist(), prob_2.tolist(), preference_2.tolist()))
                """
                gpc[j].update_post(duels, preferences);
                
                iter_end = time.time();
                iter_times[j].append(iter_end - iter_start);
                
                #cp_score[j].append(copeland(gpc[j],x, bounds, num_anchors=200, n_dims=2));

                winners[j][i] = condorcet(gpc[j], bounds, 2);


            """
            plt.figure(1);
            plt.pause(1);
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(x.numpy(), obj_f(x).numpy());
            plt.axvline(winners[1][-1], linestyle='--', color='black', label='DTS');
            plt.axvline(winners[0][-1], linestyle='--', color='red', label='Random');
            plt.legend(loc='upper left');
            plt.subplot(2,1,2);
            plt.plot(x.numpy(), cp_score[1], label='DTS');
            plt.plot(x.numpy(), cp_score[0], label='Random');

            plt.axvline(winners[1][-1], linestyle='--', color='black');
            plt.axvline(winners[0][-1], linestyle='--', color='red');
            plt.legend(loc='lower center');

            plt.show(block=False);
            """
        """
        w_rand.append(winners[0]);
        w_dts.append(winners[1]);
        times[0].append(iter_times[0]);
        times[1].append(iter_times[1]);           
        """
        end = time.time();
        
        print('Iteration {0} complete. Time taken: {1:0.3f} s'.format(k+1, end-start));
        print(winners[0][-1]);
        #break;

    cp = copeland(gpc[0], x, bounds, n_dims=2).reshape(90, 90);
    l = obj_f(x).reshape(90, 90);
    plt.subplot(1,2,1);
    plt.pcolormesh(np.unique(x[:,0].numpy()), np.unique(x[:,1].numpy()), cp.numpy(),cmap=cmap);
    start = time.time()
    w = condorcet(gpc[0], bounds, 2)
    end = time.time();
    print('Time taken to compute condorcet winner: {0:0.3f} s'.format(end-start));
    plt.scatter(w[0], w[1], color='black');
    plt.subplot(1,2,2);
    plt.pcolormesh(np.unique(x[:,0].numpy()), np.unique(x[:,1].numpy()), l.numpy(),cmap=cmap);
    plt.scatter([1],[1],color='black')
    plt.show(); 

    sys.exit();

    w_rand = torch.tensor(w_rand);
    w_dts = torch.tensor(w_dts);
    t_rand = torch.tensor(times[0]);
    t_dts = torch.tensor(times[1]);
    torch.save(w_rand, 'w_rand.pt');
    torch.save(w_dts, 'w_dts.pt');
    torch.save(t_rand, 't_rand.pt');
    torch.save(t_dts, 't_dts.pt');

    #print('Time taken to fit gp for {0} samples: {1:0.3f} s'.format(num_duels, end-start));

    start = time.time();
    cp_score = [0,0];
    cp_score[0] = copeland(gpc[0],x,bounds, 200).numpy();
    cp_score[1] = copeland(gpc[1],x,bounds, 200).numpy();
    end = time.time();

    print('Time taken to calculate copeland score: {0:0.3f} s'.format(end - start));

    start = time.time();
    winner = [0,0]
    winner[0] = condorcet(gpc[0], bounds);
    winner[1] = condorcet(gpc[1], bounds);
    end = time.time();
    print('Time taken to find Condorcet winner: {0:0.3f} s'.format(end - start)); 
    print('Error Count: {}'.format(error_count));
    plt.figure();
    plt.subplot(2,1,1)
    plt.plot(x.numpy(), obj_f(x).numpy());
    plt.axvline(winner[1], linestyle='--', color='black', label='DTS');
    plt.axvline(winner[0], linestyle='--', color='red', label='Random');
    plt.legend(loc='upper left');
    plt.subplot(2,1,2);
    plt.plot(x.numpy(), cp_score[1], label='DTS');
    plt.plot(x.numpy(), cp_score[0], label='Random');

    plt.axvline(winner[1], linestyle='--', color='black');
    plt.axvline(winner[0], linestyle='--', color='red');
    
    #plt.plot(x.numpy(), copeland(gpc,x,bounds, 200, soft=False).numpy(), label='Hard Copeland', linestyle='--');
    
    avg_w_rand = torch.mean(w_rand, dim=0);
    avg_w_dts = torch.mean(w_dts, dim=0);

    plt.legend(loc='lower center');
    plt.figure();
    plt.xlabel('Number of Samples')
    plt.ylabel('g(x_c)')
    #plt.plot(np.linspace(0,10), np.linspace(0,10));
    plt.plot((np.array(list(range(num_duels))) + 1), obj_f(avg_w_dts).numpy(), label='DTS', color='orange', linestyle='--');
    plt.plot((np.array(list(range(num_duels))) + 1), obj_f(avg_w_rand).numpy(), label='Random', color='green', linestyle='--');
    plt.legend();
    """
    

    plt.figure();
    plt.plot((np.array(list(range(num_duels))) + 1)*2, iter_times)
    """
    
    """
    n = 100
    train_x = torch.zeros(n ** 2, 2)
    train_x[:, 0].copy_(torch.linspace(0., 1., n).repeat(n))
    train_x[:, 1].copy_(torch.linspace(0., 1., n).unsqueeze(1).repeat(1, n).view(-1))
    train_y = get_preference_prob(obj_f, train_x[:, 0], train_x[:, 1]);
    
    cmap = plt.get_cmap('jet');
    fig = plt.figure();
    pcm = plt.pcolormesh(torch.linspace(0., 1., n), torch.linspace(0., 1., n), train_y.numpy().reshape(n,n), cmap=cmap)
    fig.colorbar(pcm);
    """

    #plt.figure()
    #plt.plot_prefs();
    plt.show();