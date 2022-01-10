import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Problem setup

from botorch.test_functions import Hartmann

neg_hartmann6 = Hartmann(negate=True)


def outcome_constraint(X):
    return X.sum(dim=-1)-3


def weighted_obj(X):
    return neg_hartmann6(X)* (outcome_constraint(X) <= 0).type_as(X)


from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

NOISE_SE = 0.5
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

def generate_initial_data(n=10):
    train_x = torch.rand(10,6 device=device, dtype=dtype)
    exact_obj = neg_hartmann6(train_x).unsqueeze(-1)
    exact_con = outcome_constraint(train_x).unsqueeze(-1)
    train_obj = exact_obj + NOISE_SE*torch.randn_like(exact_obj)
    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    best_observed_value = weighted_obj(train_x).max().item()
    return train_x, train_obj, train_con, best_observed_value

def initialize_model(train_x, train_obj, train_con, state_dict=None):

    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj))
    model_con = FixedNoiseGP(train_x,train_con, train_yvar.expand_as(train_con))

    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


from botorch.acquisition.objective import ConstrainedMCObjective

def obj_callable(Z):
    return Z[..., 1]

def constraint_callable(Z):
    return Z[..., 1]

