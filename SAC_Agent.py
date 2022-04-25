
"""## Soft Actor Critic

#### Spinningup Implementation
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from copy import deepcopy
import itertools
from torch.optim import Adam
import matplotlib.pyplot as plt
#plt.style.use(['science','ieee'])

#import time 
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = torch.as_tensor(act_limit, dtype=torch.float32)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [10], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPClassifierFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.z = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        z = torch.sigmoid(self.z(torch.cat([obs, act], dim=-1)))
        return torch.squeeze(z, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=((64,64),(64,64)),
                 activation=(nn.ReLU,nn.ReLU)):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes[0], activation[0], act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes[1], activation[1])
        self.lc = MLPClassifierFunction(obs_dim, act_dim, hidden_sizes[2], activation[2])

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.fail_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs,fail):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.fail_buf[self.ptr] = fail
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     fail=self.fail_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class SAC:
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    def __init__(self, env_type = "SIM", obs_dim = 4, act_dim = 1, act_limit = 1,
                 buffer_size = int(1e6), hidden_sizes = (64,64),  gamma=0.99,
                 polyak=0.995, lr=1e-3, alpha=0.2, alpha_rate=0.99,gamma_rate = 0.99,
                 activation=(nn.ReLU, nn.ReLU),seed=1):
    
        self.replay_buffer = ReplayBuffer(obs_dim = obs_dim, act_dim = act_dim, size= buffer_size)
        self.env_type = env_type
        if env_type == 'SIM':
            from USM_model import USM_model_sim
            self.env = USM_model_sim(dt=1)
            self.scale_obs = self.env.scale_obs
        
        elif env_type == 'EXP':
            from USM_model import USM_model_exp
            self.env = USM_model_exp()
            self.scale_obs = self.env.scale_obs
        
        
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.obs_dim, self.act_dim = obs_dim, act_dim    
        self.gamma, self.polyak = gamma, polyak
        self.alpha,self.alpha_rate = alpha, alpha_rate
        self.gamma_rate = gamma_rate
        self.act_limit = act_limit
        
        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(obs_dim, act_dim, act_limit, hidden_sizes = hidden_sizes, activation = activation)
        self.ac_targ = deepcopy(self.ac)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
          
        # List of parameters for both Q-networks (save this for convenience)
        #self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_params = itertools.chain(self.ac.q1.parameters())
        self.lc_params = itertools.chain(self.ac.lc.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr/2)
        self.q_optimizer = Adam(self.q_params, lr=lr)
        self.lc_optimizer = Adam(self.lc_params, lr=lr)

    
    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data):
        o, a, r, o2 = data['obs'], data['act'], data['rew'], data['obs2']
        q1 = self.ac.q1(o,a)
        #q1 = torch.mean(q1)
        #q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            #q2_pi_targ = self.ac_targ.q2(o2, a2)
            #q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            q_pi_targ = torch.min(q1_pi_targ,dim=1).values
            #print(q1_pi_targ.shape,q_pi_targ.shape,logp_a2.shape)
            #backup = r.unsqueeze(dim=1) + self.gamma * (q1_pi_targ - self.alpha * logp_a2.unsqueeze(dim=1))
            backup = r + self.gamma * (q_pi_targ - self.alpha * logp_a2)

            backup = torch.reshape(backup,(-1,1))

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        #loss_q2 = ((q2 - backup)**2).mean()
        #loss_q = loss_q1 + loss_q2
        loss_q = loss_q1

        # Useful info for logging
        #q_info = dict(Q1Vals=q1.detach().numpy(),
        #              Q2Vals=q2.detach().numpy())
        #q_info = dict(Q1Vals=loss_q.detach().numpy())
        return loss_q, loss_q.detach().numpy()

    def compute_loss_lc(self,data):
        o, a, fail = data['obs'], data['act'], data['fail']
        y_pred = self.ac.lc(o,a)
        w = torch.abs(fail - 0.1)
        loss_lc = -torch.multiply(w,(torch.multiply(fail,torch.log(y_pred + 1e-5)) + torch.multiply((1-fail),torch.log(1-y_pred +1e-5)))).mean()
        #print(loss_lc)
        #lossLC_info = dict(Q1Vals=loss_lc.detach().numpy())
      
        return loss_lc,loss_lc.detach().numpy()
        
    def compute_loss_pi(self,data):
        o = data['obs']
        #o2 = data['obs2']

        pi, logp_pi = self.ac.pi(o)
        
        #L_pi = torch.min(L1_pi, L2_pi)
        q1_pi = self.ac.q1(o, pi)
        #fail = self.ac.lc(o,pi)

        #q2_pi = self.ac.q2(o, pi)
        #q_pi = torch.min(q1_pi, q2_pi)
        #err(-)*a(-) = + 

        q_pi = torch.min(q1_pi,dim=1).values #+ 0.1*torch.sign(torch.multiply(pi,o[:,-1]) + 1e-5) #+ (1/(fail-1.02) +1/1.02) #+ 1*torch.sign(torch.multiply(pi,o[:,-1]) + 1e-5) #- 5*(torch.clip(2*fail-1,min=0))# + 0.5*torch.sign(torch.multiply(pi,o[:,-1]) + 1e-5) torch.log10(torch.clip(1-2*fail,min=0))
        #print("Q: {} -- logpi:  {} ".format(q_pi,logp_pi))
        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        #pi_info = dict(LogPi=loss_pi.detach().numpy())

        return loss_pi, loss_pi.detach().numpy()

    def update(self,data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, self.q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        self.lc_optimizer.zero_grad()
        loss_lc, self.lc_info = self.compute_loss_lc(data)
        loss_lc.backward()
        self.lc_optimizer.step()
    # Set up function for computing SAC pi loss

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False
        for p in self.lc_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, self.pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
        for p in self.lc_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q1.parameters(), self.ac_targ.q1.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
                
    def update_alpha(self):
        self.alpha *= self.alpha_rate
        self.alpha = max(0.01,self.alpha)

    def update_gamma(self):
        self.gamma *= self.gamma_rate
        self.gamma = max(0.1,self.gamma)
        #self.gamma = np.random.random()

    def get_alpha(self):
        return self.alpha
    
    def get_action(self,o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)
    def get_q(self,o):
      o = torch.as_tensor(o, dtype=torch.float32)
      a = self.get_action(o)
      a = torch.as_tensor(a, dtype=torch.float32)
      q1_ = self.ac.q1(o, a)
      #q2_ = self.ac.q2(o, a)
      #q_ = torch.min(q1_, q2_)
      q_ = torch.max(q1_)
      return q_.detach().numpy()
    
    
    def train_agent(self, batch_size=128, start_steps=-1, max_ep_len=30,
                    num_eps=1000, update_every=1, alpha_rate=0.995, eval_every = 250):
    
        # Random seed for numpy and torch
        #env.start()
        # Prepare for interaction with environment
        total_steps = max_ep_len * num_eps
        ep = 1
        
        o,ep_ret, ep_len = self.env.reset(), 0, 0 
        LOGs = []
        losses = []
        #states = []
        #actions = []
        o = self.scale_obs(o)
        q0 = self.get_q(o)
        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > start_steps:
                a = self.get_action(o)
            else:
                a = 2*(np.random.random(self.act_dim) -0.5)*self.act_limit
      
            # Step the env
            o2, r, fail = self.env.step_frequency(a)

            #states.append(o2)
            #actions.append(a)
            o2 = self.scale_obs(o2)
            ep_ret += r
            ep_len += 1

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2,fail)
      
            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            #o = self.env.update_temp()
            #o = self.scale_obs(o)
            o = o2

            # Update handling
            if t % update_every == 0:
                #env.stop()
                for j in range(update_every):
                    batch = self.replay_buffer.sample_batch(batch_size)
                    self.update(data=batch)
                    losses.append([self.q_info,self.lc_info,-self.pi_info])    

                #env.start()
            # End of trajectory handling
            if (ep_len == max_ep_len):
                #plot_eval(np.array(states),np.array(actions))
                #states=[]
                #actions=[]
                if ep % eval_every == 0:
                    self.run_evals_ep(ep,freq=39)
                if ep % 10 == 0:
                  print("Episode: {} --> Temp: {} --> TargetSpeed/Speed: {}/{} --> Torque: {}--> reward: {} --> Q0: {}".format(ep,self.env.get_temp(),self.env.get_targetSpeed(), self.env.get_speed(), self.env.get_torque(),ep_ret,q0))
                ep += 1
                LOGs.append(np.array([ep,self.env.get_temp(),self.env.get_torque(),self.env.get_targetSpeed(),ep_ret,self.env.get_speed(),q0]))
                #self.env.set_noise(np.abs(np.random.normal())*50,np.abs(np.random.normal())*0.05)
                o, ep_ret, ep_len = self.env.reset(ep=ep), 0, 0 # targetSpeed=targetSpeeds[ep]
                o = self.scale_obs(o)
                q0 = self.get_q(o)
                self.update_alpha()
                self.update_gamma()
                
        return np.array(LOGs),np.array(losses)
    
    def load_network(self, env_type='sim'):
        self.ac.pi.load_state_dict(torch.load('pi_' + env_type))
        self.ac.q1.load_state_dict(torch.load('q1_' + env_type))
        self.ac.lc.load_state_dict(torch.load('lc_' + env_type))
        
    def save_network(self, env_type='sim'):
        torch.save(self.ac.pi.state_dict(),'pi_'+env_type)
        torch.save(self.ac.q1.state_dict(),'q1_'+env_type)
        torch.save(self.ac.lc.state_dict(),'lc_'+env_type)
        
    def eval_agent(self, targetSpeeds, torques=None, freq = None,temp = None):
        #env.dt = dt
        o = self.env.reset(freq=freq,temp=temp,T=torques[0], targetSpeed = targetSpeeds[0])
        states = [o]
        actions = [np.zeros(self.act_dim)]

        for i in range(len(targetSpeeds)):
          o = self.env.set_state(targetSpeed=targetSpeeds[i],torque=torques[i])
          o = self.scale_obs(o)
          a = self.get_action(o, deterministic=True)
          actions.append(a*2)
          o, _,_ = self.env.step_frequency(a)
          states.append(o)
          if i % 10 == 0:
              self.env.update_temp()

        states = np.array(states)
        actions = np.array(actions).reshape(-1,1)
        self.plot_eval(states,actions)
        return np.concatenate((states,actions),axis=-1)

    def run_evals_ep(self,ep,n = 20,freq=None,temp=None,T=0):
        #Constant Speed
        Speeds = np.arange(n+1).reshape(-1,1)
        for targetSpeed in [0,100,200,300]:
            targetSpeeds = [targetSpeed]*n
            torques = [T]*n
            speeds = self.eval_agent(targetSpeeds,torques,freq,temp)
            plt.suptitle('Speed Tracking -- Episode: ' + str(ep), y=0.95, fontsize=20,fontweight='bold')
            Speeds = np.concatenate((Speeds,speeds),axis=-1)
            #self.env.stop()
            #time.sleep(300)
            #self.env.start(amp=3,freq=42)
        np.savetxt('ep_{}_target_{}_{}.txt'.format(ep,targetSpeed,self.env_type), Speeds, delimiter=',')

    def run_evals_full(self,eval_speeds=[100,200,300],eval_torques=[0.25,0.5]):

        #Step Speed
        #targetlevels = np.linspace(0,300, 7)
        #targetlevels = np.concatenate((targetlevels,targetlevels[::-1][1:]))
        #targetSpeeds = []
        #for level in targetlevels:
         #   targetSpeeds += [level]*20 
            
        #for torque in [0,0.2,0.5]:
        #    speeds = self.eval_agent(targetSpeeds,torques=[torque]*len(targetSpeeds),freq=39,temp=20)
        #    np.savetxt('stepSpeed_300rpm_conTorque_{}Nm_{}.txt'.format(torque,self.env_type), speeds, delimiter=',')
        
        #Sinusoidal Speed
        T = 250    #total steps
        n = 5      #num cycles
        for torque in [0,0.2,0.5]:
            for targSpeed in [150]:#175,100,50,25
                targetSpeeds = targSpeed-targSpeed*np.cos(np.linspace(0,n,T)*2*np.pi)
                speeds = self.eval_agent(targetSpeeds,torques=[torque]*len(targetSpeeds),freq=39,temp=20)
                np.savetxt('sinSpeed_{}rpm_conTorque_{}Nm_{}.txt'.format(targSpeed*2,torque,self.env_type), speeds, delimiter=',')
        
        #Step Torque
        
        #targetlevels = np.linspace(0,1,11)
        #targetlevels = np.concatenate((targetlevels,targetlevels[::-1][1:]))
        #torques = []
        #for level in targetlevels:
        #    torques += [level]*20 
        #for targSpeed in [100,200,300]:    
        #    speeds = self.eval_agent([targSpeed]*len(torques),torques,freq=39,temp=20)
        #    np.savetxt('conSpeed_{}rpm_stepTorque_1Nm_{}.txt'.format(targSpeed,self.env_type), speeds, delimiter=',')
        
        #Step Torque
        
        #targetlevels = np.linspace(0,0.5,6)
        #targetlevels = np.concatenate((targetlevels,targetlevels[::-1][1:]))
        #torques = []
        #for level in targetlevels:
        #    torques += [level]*20 
        #for targSpeed in [100,200,300]:    
        #    speeds = self.eval_agent([targSpeed]*len(torques),torques,freq=39,temp=20)
         #   np.savetxt('conSpeed_{}rpm_stepTorque_0.5Nm_{}.txt'.format(targSpeed,self.env_type), speeds, delimiter=',')
        
        
        for targspeed in [100,200,300]:
            for torque in [0.5]: #0.1,0.25,0.35,
            #Sinusoidal Torque
                T = 250    #total steps
                n = 5      #num cycles   
                torques = torque-torque*np.cos(np.linspace(0,n,T)*2*np.pi)
                speeds = self.eval_agent([targspeed]*len(torques),torques,freq=39,temp=20)
                np.savetxt('conSpeed_{}rpm_sinTorque_{}Nm_{}.txt'.format(torque,targspeed,self.env_type), speeds, delimiter=',')
            
        
    
        
    def plot_training_logs(self,LOGS,losses):
        np.savetxt('logs_{}.txt'.format(self.env_type), LOGS, delimiter=',')
        np.savetxt('losses_{}.txt'.format(self.env_type), losses, delimiter=',')

        Ep_ret = LOGS[:,-3]
        avg_ret = np.convolve(Ep_ret, np.ones(5), 'valid') / 5
        Err = np.abs(LOGS[:,-2] - LOGS[:,3])
        Avg_err = np.convolve(Err, np.ones(5), 'valid') / 5
        
        plt.figure(figsize=(16,12))
        
        plt.subplot(2,2,1)
        plt.plot(Ep_ret,label="Episode Reward");
        plt.plot(avg_ret, label="Average Reward");
        plt.plot(LOGS[:,-1], label="Expected Reward");
        plt.xlabel('Episode');
        plt.ylabel('Reward');
        plt.title('Learning Curve');
        plt.legend()
        
        plt.subplot(2,2,2)
        plt.plot(LOGS[:,3],label='Target Speed');
        plt.plot(LOGS[:,-2],label='Actual Speed');
        plt.plot(Avg_err,label='Moving Average Err');
        plt.xlabel('Episode');
        plt.ylabel('Speed [rpm]');
        plt.legend()
        
        plt.subplot(2,2,3)
        plt.plot(LOGS[:,1]);
        plt.xlabel('Episode');
        plt.ylabel('USM Temperature [°C]');
        
        plt.subplot(2,2,4)
        plt.plot(LOGS[:,2]);
        plt.xlabel('Episode');
        plt.ylabel('Load Torque [N.m]');
        
        plt.figure(figsize=(16,8))
        plt.subplot(3,1,1)
        plt.plot(losses[:,0])
        plt.ylabel("Q-loss")

        plt.subplot(3,1,2)
        plt.plot(losses[:,1])
        plt.ylabel("LC-loss")


        plt.subplot(3,1,3)
        plt.plot(losses[:,2])
        plt.ylabel("Pi-loss")
        
        # plt.figure()
        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(avg_ret, 'r-')
        # ax2.plot(LOGS[:,2], 'b--')
        
        # ax1.set_xlabel('Episode')
        # ax1.set_ylabel('Reward ', color='r')
        # ax2.set_ylabel('Target Speed [rpm]', color='b')
        # ax1.set_title('Learning Curve')
        # plt.show()

    def plot_eval(self,states,actions):
        fig, axs = plt.subplots(2, 2,figsize=(16,12))
        #Plot 1
        axs[0,0].plot(states[:,3],'r',label='Speed')
        axs[0,0].plot(states[:,4],'b--',label='TargetSpeed')
        axs[0,0].plot(abs(states[:,3]-states[:,4]), 'k',label = 'Speed Error')
        axs[0,0].set_xlabel('Step',fontweight='bold',fontsize=16)
        axs[0,0].set_ylabel('Speed [rpm]',fontweight='bold',fontsize=16)
        #plt.title('Constant Speed Tracking',fontweight='bold')
        axs[0,0].legend();
    

        axs[1,0].plot(states[:,0])
        axs[1,0].set_xlabel('Step',fontweight='bold',fontsize=16);
        axs[1,0].set_ylabel('Driving Frequency [kHz]',fontweight='bold',fontsize=16);

        ax2 = axs[1,0].twinx()
        ax2.plot((actions),'r--')
        ax2.set_ylabel('Frequency Action [kHz]',fontweight='bold',fontsize=16);
        
        #Plot 2
        axs[0,1].plot(states[:,1])
        axs[0,1].set_xlabel('Step',fontweight='bold',fontsize=16);
        axs[0,1].set_ylabel('Temperature [°C]',fontweight='bold',fontsize=16);
        #plt.title('Constant Speed Tracking',fontweight='bold',);
        #plt.ylim([0,150])

        #Plot 4
        axs[1,1].plot(states[:,2])
        axs[1,1].set_xlabel('Step',fontweight='bold',fontsize=16);
        axs[1,1].set_ylabel('Torque [N.m]',fontweight='bold',fontsize=16);