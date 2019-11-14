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

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

torch.manual_seed(2);

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
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=True)
        super(VSGPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
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

    def train(self, num_steps=50):

        start = time.time();
        for i in range(num_steps):
            # Zero backpropped gradients from previous iteration
            self.optimizer.zero_grad()
            # Get predictive output
            output = self.model(self.X)
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.y)
            loss.backward()
            #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
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


def plot_cmap(fig, gpc, X, axpoints, n, cmap):
    with torch.no_grad():
        post_f = gpc.forward(X);
        mean, var = post_f.mean, post_f.variance
        pcm = plt.pcolormesh(axpoints, axpoints, var.detach().numpy().reshape(n,n), cmap=cmap)#, vmin=0., vmax=0.5);
        fig.colorbar(pcm);

def plot_points(X, selected):
    train_x = X;
    plt.scatter(train_x[:,0].numpy(), train_x[:,1].numpy(), color='yellow', s=0.8, alpha=0.5);
    plt.scatter(selected[:,0].numpy(), selected[:,1].numpy(), color='black', s=7, alpha=0.8);
    plt.scatter(selected[-1][0], selected[-1][1], color='blue', s=10, alpha=1);

def next_point_ucb(gpc, bounds):

    def func(X):
        with torch.no_grad():
            inp = torch.from_numpy(X).float().unsqueeze(0);
            #print(inp, inp.shape)
            post_f = gpc.forward(inp);
            return -1.0 * post_f.variance.numpy()

    num_repeats = 50;
    max_var = float("inf");
    max_x = float("inf");

    init_guesses = torch.distributions.uniform.Uniform(torch.tensor([bounds[0][0]]), torch.tensor([bounds[0][1]])).sample(torch.Size([num_repeats,2]))
    for x0 in init_guesses:
        res = minimize(func, x0.numpy(), bounds=bounds);
        y = func(res.x)
        if y < max_var:
            max_var = y;
            max_x = res.x;

    return torch.tensor(max_x).float().unsqueeze(0);

def UCB(gpc, X):
    with torch.no_grad():
        post_f = gpc.forward(X)

        idx = torch.argmax(post_f.variance);

        return idx;

def get_preference_prob(obj_f, x1, x2, minimum=True):
   
    diff = obj_f(x1) - obj_f(x2);
    return torch.sigmoid((int(not minimum)*2-1)*diff);

if __name__ == '__main__':
    obj_f = lambda x: (6*x - 2)**2 * torch.sin(12*x - 4)
    bounds = (0.0,1.0);
    x = torch.linspace(bounds[0], bounds[1], 100);

    gpc = GPClassifier(sparse=False, n_dims=2);

    def random_duel(bounds):
        sample1 = torch.distributions.uniform.Uniform(torch.tensor([bounds[0]]), torch.tensor([bounds[1]])).sample()
        sample2 = torch.distributions.uniform.Uniform(sample1, torch.tensor([bounds[1]])).sample()
        sample = torch.cat((sample1, sample2), dim=0)

        return sample.reshape(1,-1)

    def copeland(gpc, x, bounds, num_anchors=500, soft=True):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor([x]).float();

        y = torch.zeros_like(x);
        anchors = torch.linspace(bounds[0], bounds[1], num_anchors).reshape(-1,1);

        for i, x_ in enumerate(x):
            
            inp = x_.repeat(num_anchors, 1);

            test_x = torch.cat((inp, anchors), dim=1);
            post_f = gpc.forward(test_x);
            probs_f = gpc.likelihood(post_f);
           
            
            probs = probs_f.mean;
            preds = probs.ge(0.5).float();
            
            try:
                y[i] = (1.0/float(num_anchors)) * probs.sum() if soft else preds.sum()/len(preds);
            except IndexError:
                y = (1.0/float(num_anchors)) * probs.sum() if soft else preds.sum()/len(preds);

        return y;

    def condorcet(gpc, bounds):

        def func(X):
            return -1.0 * copeland(gpc, X, bounds)

        num_repeats = 50;
        max_cop = float("inf");
        max_x = float("inf");

        init_guesses = torch.distributions.uniform.Uniform(torch.tensor([bounds[0]]), torch.tensor([bounds[1]])).sample(torch.Size([num_repeats,1]))
        for x0 in init_guesses:
            res = minimize(func, x0.numpy(), bounds=[bounds]);
            y = func(res.x)
            if y < max_cop:
                max_cop = y;
                max_x = res.x;

        return max_x

    num_duels = 50;
    
    start = time.time();
    for i in range(num_duels):
        duel = random_duel(bounds);
        #preference = get_preference_prob(obj_f, duel[:,0],duel[:,1]).ge(0.5).float()
        preference = torch.distributions.Bernoulli(get_preference_prob(obj_f, duel[:,0],duel[:,1])).sample().float()
        
        duel_2 = torch.flip(duel, dims=(0,1));
        preference_2 = 1. - preference;

        duels = torch.cat((duel, duel_2), dim=0);
        preferences = torch.cat((preference, preference_2), dim=0)

        gpc.update_post(duels, preferences);

    end = time.time();

    print('Time taken to fit gp for {0} samples: {1:0.3f} s'.format(num_duels, end-start));

    start = time.time();
    cp_score = copeland(gpc,x,bounds, 500).numpy();
    end = time.time();

    print('Time taken to calculate copeland score: {0:0.3f} s'.format(end - start));

    start = time.time();
    winner = condorcet(gpc, bounds);
    end = time.time();
    print('Time taken to find Condorcer winner: {0:0.3f} s'.format(end - start)); 
    
    plt.subplot(2,1,1)
    plt.plot(x.numpy(), obj_f(x).numpy());
    plt.axvline(winner, linestyle='--', color='black');
    plt.subplot(2,1,2);
    plt.plot(x.numpy(), cp_score, label='Soft Copeland');

    plt.axvline(winner, linestyle='--', label='Condorcet Winner', color='black');
    #plt.plot(x.numpy(), copeland(gpc,x,bounds, 200, soft=False).numpy(), label='Hard Copeland', linestyle='--');
    plt.legend();
    

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


    
   
    
    

