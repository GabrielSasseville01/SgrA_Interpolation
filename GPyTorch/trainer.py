import gpytorch
from model import MultiTaskGPModel
import torch

class Trainer():
    def __init__(self, full_train_x, full_train_idx, full_train_y, likelihood = 'gaussian', max_early_stop = 30, iterations = 1000, verbose = False):
        self.full_train_x = full_train_x.cuda() # remove cuda if not supported
        self.full_train_idx = full_train_idx.cuda() # remove cuda if not supported
        self.full_train_y = full_train_y.cuda() # remove cuda if not supported
        self.likelihood = likelihood
        self.ll = 0
        self.model = 0
        self.optimizer = 0
        self.mll = 0
        self.max_early_stop = max_early_stop
        self.iterations = iterations
        self.verbose = verbose
    
    def initialize(self):
        if self.likelihood == 'gaussian':
            self.ll = gpytorch.likelihoods.GaussianLikelihood().cuda()
        else:
            raise ValueError('Likelihood must be gaussian')
        
        self.model = MultiTaskGPModel((self.full_train_x, self.full_train_idx), self.full_train_y, self.ll).cuda() # remove cuda if not supported

        self.model.train()
        self.ll.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.ll, self.model)

    def step(self, i):
        self.optimizer.zero_grad()
        output = self.model(self.full_train_x, self.full_train_idx)
        loss = -self.mll(output, self.full_train_y)
        loss.backward()
        
        if self.verbose:
            print(f"Iter {i + 1}/{self.iterations} - Loss: {loss.item():.3f}")

        self.optimizer.step()

        return loss

    def train_model(self):
        self.initialize()

        best_loss = 10000
        counter = 0

        for i in range(self.iterations):
            loss = self.step(i)

            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            
            if counter == self.max_early_stop:
                break
        
        return self.model, self.ll