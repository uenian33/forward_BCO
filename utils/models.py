import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent

from torch.nn.functional import gumbel_softmax
import math
ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
class forward_dynamics_continuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, uncertain=False, do_rate=0.1):
        super(forward_dynamics_continuous, self).__init__()
        
        self.linear_1 = nn.Linear(action_dim+state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, state_dim)
        self.linear_3_m = nn.Linear(hidden_dim, state_dim)
        self.linear_3_v = nn.Linear(hidden_dim, state_dim)
        
        self.uncertain = uncertain
        self.dropout = nn.Dropout(p=do_rate)

    def forward(self, s, a):
        x = torch.cat((s,a), dim=1)
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(self.dropout(x))
        x = F.leaky_relu(x, 0.001)
        if self.uncertain:
            x_m = self.linear_3_m(self.dropout(x))
            x_v = torch.exp(self.linear_3_v(self.dropout(x)))
            return x_m, x_v
        else:
            x = self.linear_3(x)
            return x
        
    def sample(self, s, a):
        x_m, x_v = self.forward(s, a)
        samples = Normal(x_m, x_v).rsample()
        return samples

class policy_multinomial(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, n_heads, do_rate=0.08):
        super(policy_multinomial, self).__init__()
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, action_dim*n_heads)
        self.n_heads = n_heads
        self.action_dim = action_dim
        self.dropout = nn.Dropout(p=do_rate)

    def forward(self, x):
        n_heads = self.n_heads
        action_dim = self.action_dim
        
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(self.dropout(x))
        x = F.leaky_relu(x, 0.001)
        x = self.linear_3(self.dropout(x))
        squeeze = False
        
        if len(x.shape) < 2:
            squeeze = True
            x = torch.unsqueeze(x, 0)
        
        uni_probs = torch.ones((x.shape[0], n_heads))
        head_samples = torch.distributions.Categorical(uni_probs).sample()
        one_hot = torch.nn.functional.one_hot(head_samples, n_heads) 
        one_hot = torch.reshape(one_hot, (one_hot.shape[0], n_heads, 1))
        
        x = torch.reshape(x, (x.shape[0], n_heads, action_dim))
        x = x * one_hot
        x = torch.sum(x, dim=1)
        
        if squeeze:
            x = torch.squeeze(x, 0)
                    
        return x
    
    def reparam_forward(self, x):
        return self.forward(x)
        
    
class policy_continuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, uncertain=False, do_rate=0.01, normalize_output=True):
        super(policy_continuous, self).__init__()
        self.normalize_output = normalize_output
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3_m = nn.Linear(hidden_dim, action_dim)
        self.linear_3_v = nn.Linear(hidden_dim, action_dim)
        self.linear_3 = nn.Linear(hidden_dim, action_dim)
        self.uncertain = uncertain
        self.dropout = nn.Dropout(p=do_rate)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(self.dropout(x))
        x = F.leaky_relu(x, 0.001)
        if self.uncertain:
            x_m = self.linear_3_m(self.dropout(x))
            if self.normalize_output:
                x_m = F.tanh(x_m)
            x_v = torch.exp(self.linear_3_v(self.dropout(x)))
            return x_m, x_v
        else:
            x = self.linear_3(self.dropout(x))
            if self.normalize_output:
                x_m = F.tanh(x_m)
            return x
        
    def sample(self, x):
        x_m, x_v = self.forward(x)
        samples = Normal(x_m, x_v).rsample()
        return samples
    
    def log_prob(self, x, pred):
        x_m, x_v = self.forward(x)
        return Independent(Normal(x_m, x_v), 1).log_prob(pred)
    


class policy_discrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(policy_continuous, self).__init__()
        self.flag_var = predict_var
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.01)
        x = self.linear_2(x)
        x = F.leaky_relu(x, 0.01)
        x = F.softmax(self.linear_3(x), dim=-1)
        return x
   
class inv_dynamics_continuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, uncertain=False, do_rate=0.2):
        super(inv_dynamics_continuous, self).__init__()
        self.linear_1 = nn.Linear(2*state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, action_dim)
        self.linear_3_m = nn.Linear(hidden_dim, action_dim)
        self.linear_3_v = nn.Linear(hidden_dim, action_dim)
        self.uncertain = uncertain
        self.dropout = nn.Dropout(p=do_rate)

    def forward(self, s, s_prime):
        x = torch.cat([s, s_prime], dim=1)
        x = self.linear_1(self.dropout(x))
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(self.dropout(x))
        x = F.leaky_relu(x, 0.001)
        if self.uncertain:
            x_m = self.linear_3_m(self.dropout(x))
            x_v = torch.exp(self.linear_3_v(self.dropout(x)))
            return x_m, x_v
        else:
            x = self.linear_3(self.dropout(x))
            return x
        
    def sample(self, x):
        x_m, x_v = self.forward(x)
        samples = Normal(x_m, x_v).rsample()
        return samples

class inv_dynamics_discrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(inv_dynamics_discrete, self).__init__()
        self.linear_1 = nn.Linear(2*state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, s, s_prime):
        x = torch.cat([s, s_prime], dim=1)
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.01)
        x = self.linear_2(x)
        x = F.leaky_relu(x, 0.01)
        x = F.softmax(self.linear_3(x), dim=-1)
        return x


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, n_hidden,  num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.feature_layers = nn.Sequential(
                                nn.Linear(in_features, n_hidden),
                                nn.Tanh(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.Tanh(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.Tanh(),
        )
        self.pi = nn.Sequential(
            nn.Linear(n_hidden, num_gaussians),
        )
        self.sd = nn.Linear(n_hidden, out_features * num_gaussians)
        self.mu = nn.Linear(n_hidden, out_features * num_gaussians)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, minibatch, use_logit=True):
        x = self.feature_layers(minibatch)
        if use_logit:
            pi = self.softmax(self.pi(x))
        else:
            pi = self.pi(x)
            
        sigma = torch.exp(self.sd(x))#torch.exp(self.sigma(minibatch))
        #print(sigma.shape)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu
    
    def reparam_forward(self, x, tau=10e-10):
        pi, sigma, mu = self.forward(x, use_logit=False)
        onehot_k = gumbel_softmax(torch.tensor(pi), tau=tau, eps=1e-30)
        onehot_k = onehot_k.unsqueeze(1)
        #print(pi.shape)
        #print(onehot_k.shape)
        y = torch.distributions.Normal(mu,sigma).rsample()
        y = y.permute(0,2,1)
        #print(onehot_k.shape, y.shape)
        y = (y*onehot_k)
        y = torch.sum(y, dim=2)
        
        return y
    
    
    def mdn_loss(self, inputs, target):
        """Calculates the error, given the MoG parameters and the target

        The loss is the negative log likelihood of the data given the MoG
        parameters.
        """
        pi, sigma, mu = self.forward(inputs)
        prob = pi * self.gaussian_probability(sigma, mu, target)
        nll = -torch.log(torch.sum(prob, dim=1))
        #if torch.isnan(torch.mean(nll)):
        #    print(sigma)
        return torch.mean(nll)
    
    def mdn_sample(self, pi, sigma, mu):
        """Draw samples from a MoG.
        """
        # Choose which gaussian we'll sample from
        pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
        # Choose a random sample, one randn for batch X output dims
        # Do a (output dims)X(batch size) tensor here, so the broadcast works in
        # the next step, but we have to transpose back.
        gaussian_noise = torch.randn(
            (sigma.size(2), sigma.size(0)), requires_grad=False)
        variance_samples = sigma.gather(1, pis).detach().squeeze()
        mean_samples = mu.detach().gather(1, pis).squeeze()
        return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
 
    def gaussian_probability(self, sigma, mu, target):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.

        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.unsqueeze(1).expand_as(sigma)
        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
        return torch.prod(ret, 2)
