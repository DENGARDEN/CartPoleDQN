import torch

from . import *
from .memory import ReplayMemory
from .model import SimpleMLP

from collections import deque
import random
import statistics


class DQNTrainer:
    def __init__(
            self,
            config
    ):
        self.config = config
        self.env = gym.make(config.env_id)
        self.epsilon = self.config.eps_start  # My little gift for you

        # self.optimizer = self.config.optim_cls(, **self.config.optim_kwargs)
        self.q = SimpleMLP(input_size=4, output_size=2, hidden_sizes=[64, ])  # s,a,r,s*
        self.q_fixed = SimpleMLP(input_size=4, output_size=2, hidden_sizes=[64, ])
        self.q_fixed.load_state_dict(self.q.state_dict())
        self.optimizer = self.config.optim_cls(self.q.parameters(), lr=self.config.optim_kwargs['lr'])

        self.replay_buffer = ReplayMemory(num_steps=self.config.n_steps)

    def train(self, num_train_steps: int):

        i = 0
        self.trained_steps = i
        episode_rewards = []
        
        # Whatever your train loop may be.
        while i < num_train_steps:
            begin_steps = self.trained_steps
            
            state = self.env.reset()
            done = False
            score = 0

            self.env.render()


            # make an episode
            while not done:

                if i != 0:
                    self.update_epsilon()
                action = self.predict(ob=torch.from_numpy(state).float())
                next_state, reward, done, info = self.env.step(action)
                done_tensor_mask = 0.0 if done else 1.0
                self.replay_buffer.write(state=state, action=action, reward=reward / 100.0, next_state=next_state,
                                         done_mask=done_tensor_mask)

                score += reward
                state = next_state

                i += 1

                if self.replay_buffer.__len__() > self.config.learning_starts:
                    self.update_network()

                if i % self.config.target_update_freq == 0 or done:
                    self.update_target()

                if done:
                    episode_reward = i - begin_steps

                    # if self.config.verbose:
                    #     status_string = f"{self.config.run_name:10}, Whatever you want to print out to the console"
                    #     print(status_string + "\r", end="", flush=True)

                    # episode_rewards.append(episode_reward)
                    episode_rewards.append(score)
                    self.trained_steps = i
                    if len(episode_rewards) > 10:
                        print(
                            f"n_episode : {len(episode_rewards)}, recent_mean_reward : {statistics.mean(episode_rewards[-10:-1])}")

                    break

        return episode_rewards

    # Update online network with samples in the replay memory. 
    def update_network(self):
        for i in range(self.config.train_freq):
            if self.config.n_steps == 1:
                """TD(0)"""
                s, a, r, next_s, done_mask = self.replay_buffer.sample(self.config.batch_size)
                output = self.q(s)
                q_a = output.gather(1, a)  # fetch q value from state-action pair
                target = r + self.config.discount_rate * (self.q_fixed(next_s).max(1)[0].unsqueeze(1)) * done_mask
            else:
                """TD(n)"""
                # discounted sum; r
                s, a, r, next_s, done_mask = self.replay_buffer.multi_step_sample(self.config.batch_size,
                                                                                  discount=self.config.discount_rate,
                                                                                  n_step=self.config.n_steps)
                output = self.q(s)
                q_a = output.gather(1, a)  # fetch q value from state-action pair
                target = r + (self.config.discount_rate ** self.config.n_steps) * (
                    self.q_fixed(next_s).max(1)[0].unsqueeze(1)) * done_mask
                # s, a, r, next_s, done_mask = self.replay_buffer.sample(self.config.batch_size)  # tensor form...
                # output = self.q(s)
                # q_a = output.gather(1, a)  # fetch q value from state-action pair
                # target = r
                #
                # rewards = torch.zeros(r.size())
                # # states = deque(maxlen=self.config.n_steps - 1)
                # # done_list = deque(maxlen=self.config.n_steps - 1)
                #
                # # for step in range(self.config.n_steps - 1):
                # #     if (len(states) == 0):
                # #         """for finding R2"""
                # #         action = self.predict(ob=next_s)
                # #     else:
                # #         """for finding R3 <= """
                # #         action = self.predict(ob=states[step - 1])
                # #     next_state, reward, done, info = self.env.step(action)
                # #     rewards.append(reward)
                # #     states.append(next_state)
                # #     done_list.append(0.0 if done else 1.0)
                # for states in s:
                #     index = self.replay_buffer.memory.index(states)
                #     for cnt in range(self.config.n_steps - 1):
                #         rewards += (self.config.discount_rate ** (cnt + 1) * self.replay_buffer.memory[
                #             index + cnt + 1].reward * (
                #                         self.replay_buffer.memory[index].done_mask))
                #
                #     rewards += (self.config.discount_rate ** (self.config.n_steps) * (
                #         self.q_fixed(self.replay_buffer.memory[index + self.config.n_steps].state).max(1)[0].unsqueeze(
                #             1)) * self.replay_buffer.memory[index + self.config.n_stpes].done_mask)

                # for reward in rewards:
                #     target += reward
            # batch_transition = self.replay_buffer.sample(self.config.batch_size)
            # output = self.q(torch.FloatTensor(batch_transition.state))  # outputs tensor
            # q_a = output.gather(dim=1, index=torch.tensor(batch_transition.action).type(torch.int64).unsqueeze(-1))
            # target = batch_transition.reward + self.config.discount_rate * (self.q_fixed(
            #     batch_transition.next_state).max()) if not batch_transition.done else batch_transition.reward

            loss = self.config.loss_fn(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Update the target network's weights with the online network's one.
    def update_target(self):
        self.q_fixed.load_state_dict(self.q.state_dict())
        print(f"n_steps : {self.trained_steps}, epsilon : {self.epsilon}")

    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(self, ob):
        output = self.q.forward(ob)
        if random.random() < self.epsilon:
            return random.randint(0, 1)  # outputs random action
        else:
            return output.argmax().item()

    # Update epsilon over training process.
    def update_epsilon(self):
        self.epsilon = max(self.config.eps_end, self.epsilon * self.config.eps_decay)
