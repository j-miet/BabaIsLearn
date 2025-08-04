"""Implements Reinforce class"""

import datetime
import copy
import os
from pathlib import Path
import time

from pynput.keyboard import Key
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F

from algorithms.reinforce.policy import Policy
from algorithms.reinforce.value import ValueNet
from api.babagame import BabaGame
from api.hotkey import Hotkey


class Reinforce:
    """REINFORCE algorithm with baseline value network."""
    def __init__(self, 
                game: BabaGame, 
                policy_lr: float=0.001,
                value_lr: float=0.01,
                gamma: float=0.99,
                episodes: int=200,
                steps: int=200,
                step_reward: int | float=0,
                failed_reward: int | float=0,
                success_reward: int | float=1,
                score: int | float=0,
                baseline_enabled: bool=True,
                models_path: Path=Path(__file__).parent/'models',
                policy_file: str='reinforce_policy.pt',
                value_file: str='reinforce_value.pt',
            ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\n> Selected device:", self.device)

        self.game = game
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.gamma = gamma # rewards discount factor
        self.episodes = episodes
        self.steps = steps
        self.step_reward = step_reward
        self.failed_reward = failed_reward # if steps limit is reached and map has not been won
        self.success_reward = success_reward
        self.score = score # starting score for a new episode
        self.baseline_enabled = baseline_enabled
        self.models_path = models_path
        self.policy_file = policy_file
        self.value_file = value_file

        self._undo_penalty = 0 # penalty (= negative reward) after selecting 'undo' action
        self._input_delay = 0.02 # must be a small value, but not too small or pynput cannot register basic inputs
        self._input_delay2 = 1.5
        self._game_status = 0
        self._win_counter = 0
        self._episode_loss = 0

        self.policy = Policy(action_count=4).to(self.device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.policy_lr)
        if self.baseline_enabled:
            self.value_net = ValueNet().to(self.device)
            self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=self.value_lr)

        # visualization with matplotlib
        self._scores = []
        _, self.axs = plt.subplots(2, figsize=(7,6))
        plt.subplots_adjust(hspace=0.5)
        self.action_names = ["Up", "Down", "Left", "Right"] # "Wait" & "Undo" are unused
        self.labels = [0 for _ in range(len(self.action_names))]
        self.axs[0].set_ybound(0,2) # adjust this based on total rewards range
        self.axs[0].set_title('Episode rewards')
        self.axs[0].set_ylabel('Reward')
        self.axs[0].set_xlabel('Episode')
        self.axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        self.bar = self.axs[1].bar(self.action_names, self.labels, width=0.3)
        self.axs[1].set_ybound(0,1)
        self.axs[1].set_yticks([0.2*y for y in range(0,len(self.action_names))])
        self.axs[1].set_title('Action probabilities')
        self.axs[1].set_ylabel('Probability')
        plt.ion()
        plt.show()

    def _select_action(self, state_list: list[list[int]]):
        """Sample an action based on given state.
        
        The list which is created under baba_api.py is converted into a pytorch tensor, then forwarded through a policy 
        network. Network uses softmax to produce a valid tensor, which is used to create a categorical distribution. Action 
        is sampled from this distribution and its probability is appended into network's list of probabilities.

        Finally, return the index corresponding to sampled action (an integer 0-5)
        """
        flat_state = [xy for row in state_list for xy in row]
        state = torch.tensor(flat_state, dtype=torch.float64).to(self.device, dtype=torch.float)
        policy = self.policy(state)
        m = Categorical(policy)
        action = m.sample()
        self.policy.log_probs.append(m.log_prob(action).reshape(1))
        self.policy.entropies.append(m.entropy())
        self.labels = m.probs.tolist() # save probabilities for plotting
        if self.baseline_enabled:
            self.value_net.values.append(self.value_net(state)) # forward the value network
        return action.item()

    def _press_key(self, action: int) -> None:
        """Selects and presses keyboard input based on given action.
        
        Supports two keyboard layouts to speed up action input speed and thus learning process.
        Keys can be changed in api.hotkey.py
        """
        key: Key | str
        if Hotkey.pressed:
            match action:
                case 0:
                    key = Hotkey.UP1
                case 1:
                    key = Hotkey.DOWN1
                case 2:
                    key = Hotkey.LEFT1
                case 3:
                    key = Hotkey.RIGHT2
                case 4:
                    key = Hotkey.WAIT1
                case 5:
                    key = Hotkey.UNDO1
            Hotkey.pressed = 0
        else:
            match action:
                case 0:
                    key = Hotkey.UP2
                case 1:
                    key = Hotkey.DOWN2
                case 2:
                    key = Hotkey.LEFT2
                case 3:
                    key = Hotkey.RIGHT2
                case 4:
                    key = Hotkey.WAIT2
                case 5:
                    key = Hotkey.UNDO2
            Hotkey.pressed = 1
        Hotkey.keyboard.press(key)
        time.sleep(self._input_delay)
        Hotkey.keyboard.release(key)
        time.sleep(self._input_delay)
        self.policy.actions.append(action)

    def _step_env(self, action: int) -> tuple[list[list[int]], int, int]:
        """Performs a single step into environment.
        
        Returns next state, reward, and identifier if current state is terminal (level was won) or not.
        """
        self._press_key(action)
        state_val = self.game.get_state()
        reward = 0
        if state_val[1] == 1:
            if action == 5: # penaltize 'undo' action
                reward = self._undo_penalty
            else:
                reward = self.step_reward
        elif state_val[1] == 2:
            reward = self.success_reward
        else:
            return [[]], 0, -1
        return state_val[0], reward, state_val[1]

    def _train(self) -> None:
        """Perform a single training step by using REINFORCE algorithm with baseline state-value network.
        
        REINFORCE with baseline:  
        G = reward  
        GAMMA = reward discount factor  
        v(S, w) = state-value  
        DELTA = advantage  
        w = value net parameter  
        THETA = policy net parameter  
        a_w, a_THETA = learning rates/step sizes  
        Grad[] = gradient

        Update process:
        DELTA <-- G - v(S, w)
        w <-- w + a_w * GAMMA^t * DELTA * Grad[v(S, w)]
        THETA <-- THETA + a_THETA * GAMMA^t * DELTA * Grad[log(policy)]
        """
        r = 0
        v = 0
        returns_list = []
        values_list = []

        for reward in self.policy.rewards[::-1]:
            r = reward + self.gamma*r
            returns_list.insert(0, r)
        returns = torch.tensor(returns_list)

        if self.baseline_enabled:
            for val in self.value_net.values[::-1]:
                v = val + self.gamma*v
                values_list.insert(0, r)
            values = torch.tensor(values_list)
            advantages = (returns-values).to(self.device)
            value_loss = F.mse_loss(values, returns)
        else:
            advantages = returns.to(self.device)

        log_probs = torch.cat(self.policy.log_probs).to(self.device)
        entropy_loss = -torch.mean(torch.tensor(self.policy.entropies))
        policy_loss = -torch.mean(log_probs * advantages)
        policy_loss = policy_loss + 0.001 * entropy_loss 
        self._episode_loss = policy_loss.item()

        # train value network
        if self.baseline_enabled:
            self.value_optimizer.zero_grad()
            value_loss.requires_grad = True
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float('inf'))
            self.value_optimizer.step()

        # train policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1)
        self.optimizer.step()

        self.value_net.values.clear()
        self.policy.log_probs.clear()
        self.policy.rewards.clear()
        self.policy.entropies.clear()

    def begin(self) -> None:
        """Begin a training loop."""
        global_step = 0
        self._game_status = 0
        self._win_counter = 0

        if os.path.isfile(Path(__file__).parent/f"{self.policy_file}"):
            self.policy.load_state_dict(torch.load(self.models_path/f"{self.policy_file}", weights_only=True))
            print("Loaded existing policy model.")
        else:
            print("No existing policy network found.")
        if self.baseline_enabled:
            if os.path.isfile(Path(__file__).parent/f"{self.value_file}"):
                self.value_net.load_state_dict(torch.load(self.models_path/f"{self.value_file}", 
                                                    weights_only=True))
                print("Loaded existing value model.")
            else:
                print("No existing value network found.")
        Hotkey.press_space()
        time.sleep(self._input_delay2)
        Hotkey.press_space()
        time.sleep(self._input_delay2)
        print("Starting a new step training loop...")
        start_time = time.time()

        for e in range(1, self.episodes+1):
            if Hotkey.stopped:
                print("Learning process halted.")
                Hotkey.stopped = 0
                break
            elif self._game_status == 1:
                Hotkey.keyboard.press('r')
                time.sleep(self._input_delay)
                Hotkey.keyboard.release('r')
                time.sleep(self._input_delay2)
            elif self._game_status == -1:
                print("Game closed.")
                break
            elif self._game_status == 2:
                print("Replaying the previous map.")
                time.sleep(self._input_delay2)
                Hotkey.press_space()
                time.sleep(self._input_delay2)
                Hotkey.press_space()
                time.sleep(self._input_delay2)
                Hotkey.press_space()
                time.sleep(self._input_delay2)

            self.game.flush_stdout() # flush stdout to erase previously queued output
            state, status = self.game.get_state() # enter a map and receive initial state
            if status == -1:
                self._game_status == -1
                break
            step = 1
            score = self.score
            while step < self.steps+1:
                self._game_status = 0
                
                action = self._select_action(state)
                next_state, reward, self._game_status = self._step_env(action)
                if step == self.steps and self._game_status != 2:
                    reward = self.failed_reward
                
                self.policy.rewards.append(reward)
                score += reward
                state = copy.deepcopy(next_state)

                if self._game_status == 2:
                    self._win_counter += 1
                    break # victory
                for i in range(len(self.action_names)):
                    self.bar[i].set_height(self.labels[i])
                plt.pause(0.001)

                global_step += 1
                step += 1
            
            # train model, update rewards graph and print statistics
            self._scores.append(score)
            self._train()
            self.axs[0].plot(range(1,e+1), self._scores)
            plt.pause(0.001)

            print(
                f'Episode {e} -> reward: {score} | global step: {global_step} '
                f'/// wins: {self._win_counter}, time total: {datetime.timedelta(seconds=int(time.time()-start_time))}'
            )
            self._episode_loss = 0
            self.policy.actions.clear()
            
        self._scores.clear()
        if self._game_status == -1:
            print("Any updates to policy network were not saved.")
            return
        torch.save(self.policy.state_dict(), self.models_path/f"{self.policy_file}")
        torch.save(self.value_net.state_dict(), self.models_path//f"{self.value_file}")
        print(f"Model parameters saved.")