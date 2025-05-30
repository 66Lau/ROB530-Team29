# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
from torch.distributions import Normal, Beta

class ActorCriticMLP(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        init_noise_std=1.0,
                        activation = 'elu',
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticMLP, self).__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        num_actions *= 2
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                actor_layers.append(activation)
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        print(actor_layers)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        glide_critic_layers = []
        glide_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        glide_critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                glide_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                glide_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                glide_critic_layers.append(activation)
        self.glide_critic = nn.Sequential(*glide_critic_layers)

        push_critic_layers = []
        push_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        push_critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                push_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                push_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                push_critic_layers.append(activation)
        self.push_critic = nn.Sequential(*push_critic_layers)

        reg_critic_layers = []
        reg_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        reg_critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                reg_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                reg_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                reg_critic_layers.append(activation)
        self.reg_critic = nn.Sequential(*reg_critic_layers)


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Glide Critic MLP: {self.glide_critic}")
        print(f"Push Critic MLP: {self.push_critic}")
        print(f"Reg Critic MLP: {self.reg_critic}")

        # Action noise
        self.std = torch.ones(num_actions)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        Beta.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        if (torch.isnan(observations).any()):
            print("!!!obs==none!!!")
        action = self.actor(observations)
        if (torch.isnan(action).any()):
            print("!!!act==none!!!")
        alpha = torch.clamp(action[:, 0::2], min=1e-6)
        beta = torch.clamp(action[:, 1::2], min=1e-6)
        # self.distribution = Normal(mean, mean*0. + 0.8)
        self.distribution = Beta(alpha, beta)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample() * 6 - 3
    
    def get_actions_log_prob(self, actions):
        actions = torch.clamp(actions, min=-3+1e-6, max=3-1e-6)
        original_actions = (actions + 3) / 6.0
        log_prob = self.distribution.log_prob(original_actions).sum(dim=-1)
        log_prob -= torch.log(torch.tensor(6.0))
        return log_prob

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        alpha = torch.clamp(actions_mean[:, 0::2], min=1e-6)
        beta = torch.clamp(actions_mean[:, 1::2], min=1e-6)
        return (alpha/(alpha + beta))*6 - 3

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        glide_value = self.glide_critic(critic_observations)
        push_value = self.push_critic(critic_observations)
        reg_value = self.reg_critic(critic_observations)
        return value, glide_value, push_value, reg_value
    
class SoftplusWithOffset(nn.Module):
    def __init__(self, beta=1, threshold=20):
        super(SoftplusWithOffset, self).__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        return self.softplus(x) + 1 + 1e-6
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "softplusOffset":
        return SoftplusWithOffset()
    else:
        print("invalid activation function!")
        return None