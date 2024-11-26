import gpytorch

class MultiTaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultiTaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.RBFKernel()

            # Learn an IndexKernel for 2 tasks
            self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

        def forward(self, x, i):
            mean_x = self.mean_module(x)

            # Input-input covariance
            covar_x = self.covar_module(x)
            # Task-task covariance
            covar_i = self.task_covar_module(i)
            # Combined covariance
            covar = covar_x.mul(covar_i)

            return gpytorch.distributions.MultivariateNormal(mean_x, covar)