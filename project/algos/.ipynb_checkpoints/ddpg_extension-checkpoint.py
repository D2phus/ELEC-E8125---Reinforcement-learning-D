from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
import torch.nn as nn
from pathlib import Path

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGExtension(DDPGAgent):
    def __init__(self, config=None, 
                 noise: float=0.1, noise_clip: float=0.3, policy_update_freq: int=2,
                 
                ):
        """
        noise: for action exploration. 
        noise_clip: clip the noise, serving as a regularizer
        policy_update_freq: delayed policy update.
        cur_num_iter: current number of iterations for controlling policy update.
        """
        super(DDPGExtension, self).__init__(config)
        #### TD3
        self.noise = noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq
        self.cur_num_iter = 0 
        
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)

        if not evaluation and self.buffer_ptr < self.random_transition: 
            # collect random trajectories for better exploration.
            # HINT: should not use when evalution
            action = torch.rand(self.action_dim) * self.max_action # NOTE the action range. 
        else:
            expl_noise = self.noise * self.max_action # the stddev of the expl_noise if not evaluation
            action = self.pi(x) # deterministic action 
            
            if not evaluation: # if evaluation equals False, add normal noise to the action. 
                action += torch.clamp(expl_noise*torch.randn(action.shape), min=-self.noise_clip, max=self.noise_clip) #### TD3: clamp the noise

        return action, {} # just return a positional value
    
    def train_iteration(self):
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action, act_logprob = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)     
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            # record the transitions to the buffer 
            self.record(obs, action, next_obs, reward, done_bool)
            
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()
        
        #### TD3: update critic and policy (delayed)
        self.cur_num_iter += 1
        info = self.update()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
        
    
    def _update(self,):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        
        # batch contain
        state = batch.state.to(self.device) # [batch, state_dim]
        action = batch.action.to(self.device) # [batch, action_dim]
        next_state = batch.next_state.to(self.device) # [batch, state_dim]
        reward = batch.reward.to(self.device) # [batch, 1]
        not_done = batch.not_done.to(self.device) # [batch, 1]
        
        
        # compute Q target with q_target and pi_target networks 
        with torch.no_grad():
            next_action = self.pi_target(next_state) 
            #### TD3: clipped double-Q learning: compute double Q-values for the next state, and use the smaller one for target Q calculation.
            next_q_1 = self.q_target(next_state, next_action) # (batch_size, num_quant)
            next_q_2 = self.q_target(next_state, next_action)
            next_q  = torch.min(next_q_1, next_q_2)

        target_q = reward + self.gamma * not_done * next_q # NOTE that Q value will be zero when reaching the final state 
        
        
        #### quantile, Huber loss
        # compute critic loss
        current_q = self.q(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        #### TD3: delayed policy updates
        if self.cur_num_iter % self.policy_update_freq == 0:
            # compute actor loss
            actor_loss = -torch.mean(self.q(state, self.pi(state)))  # NOTE negative: maximum 
            # optimize the actor
            self.pi_optim.zero_grad()
            actor_loss.backward()
            self.pi_optim.step()

            # update the target q and target pi using cu.soft_update_params() function
            cu.soft_update_params(self.q, self.q_target, self.tau)
            cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}