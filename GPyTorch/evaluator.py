import gpytorch
import torch

class Evaluator():
    def __init__(self, model, ll, test_x, test_idx, num_keys = 4):
        self.model = model
        self.ll = ll
        self.test_x = test_x
        self.test_idx = test_idx
        self.num_keys = num_keys
        self.observed_preds = []

    def initialize(self, model, ll):
        model.eval()
        ll.eval()

    def predict(self, test_x, test_idx, num_keys):
        # Make predictions - one task at a time using the specific test points
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(num_keys):
                # Predictions for all tasks
                self.observed_preds.append(self.ll(self.model(test_x[i].cuda(), test_idx[i].cuda()))) # remove cuda if not supported

    def evaluate(self):
        self.initialize(self.model, self.ll)
        self.predict(self.test_x, self.test_idx, self.num_keys)

        return self.observed_preds