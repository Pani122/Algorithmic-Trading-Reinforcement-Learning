import numpy as np
import math
import gym
from gym import spaces
from random import random

class TradingEnvironment(gym.Env):
    # Constants
    PENALTY_FACTOR = 1.0

    def __init__(self, data_dir, target_symbols, input_symbols, start_date, end_date, window_size=60, stop_loss=-1.0, use_cumulative_reward=False):
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.stop_loss = stop_loss
        self.use_cumulative_reward = use_cumulative_reward
        self.target_symbols = target_symbols
        self.input_symbols = []
        self.data_store = {}

        # Load data for each symbol
        for symbol in (target_symbols + input_symbols):
            file_path = f"{data_dir}/{symbol}.csv"
            symbol_data = {}
            last_close = 0
            last_volume = 0

            try:
                with open(file_path, "r") as file:
                    for line in file:
                        if line.strip():
                            dt, open_price, high, low, close, volume = line.strip().split(",")
                            try:
                                if dt >= start_date:
                                    high = float(high) if high else float(close)
                                    low = float(low) if low else float(close)
                                    close = float(close)
                                    volume = int(volume)

                                    if last_close > 0 and close > 0 and last_volume > 0:
                                        close_change = (close - last_close) / last_close
                                        high_change = (high - close) / close
                                        low_change = (low - close) / close
                                        volume_change = (volume - last_volume) / last_volume
                                        symbol_data[dt] = (high_change, low_change, close_change, volume_change)

                                    last_close = close
                                    last_volume = volume
                            except Exception as e:
                                print(f"Error parsing line: {line.strip().split(',')}\nException: {e}")

                # Store data if enough records exist
                if len(symbol_data) > window_size:
                    self.data_store[symbol] = symbol_data
                    if symbol in target_symbols:
                        self.target_symbols.append(symbol)
                    if symbol in input_symbols:
                        self.input_symbols.append(symbol)
            except Exception as e:
                print(f"Error loading file: {file_path}\nException: {e}")

        self.actions = ["LONG", "SHORT"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(np.ones(window_size * (len(input_symbols) + 1)) * -1, np.ones(window_size * (len(input_symbols) + 1)))

        self.reset()
        self._seed()

    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.reward = 0
        if self.actions[action] == "LONG":
            if sum(self.positions) < 0:
                for pos in self.positions:
                    self.reward += -(pos + 1)
                if self.use_cumulative_reward:
                    self.reward /= max(1, len(self.positions))
                if self.stop_loss * len(self.positions) > self.reward:
                    self.done = True
                self.positions = []
            self.positions.append(1.0)
        elif self.actions[action] == "SHORT":
            if sum(self.positions) > 0:
                for pos in self.positions:
                    self.reward += pos - 1
                if self.use_cumulative_reward:
                    self.reward /= max(1, len(self.positions))
                if self.stop_loss * len(self.positions) > self.reward:
                    self.done = True
                self.positions = []
            self.positions.append(-1.0)

        # Update state and cumulative profit/loss
        price_change = self.current_target[self.target_dates[self.current_index]][2]
        self.cumulative_profit *= (1 + price_change)

        for i in range(len(self.positions)):
            self.positions[i] *= TradingEnvironment.PENALTY_FACTOR * (1 + price_change * (-1 if sum(self.positions) < 0 else 1))

        self.update_state()
        self.current_index += 1

        if self.current_index >= len(self.target_dates) or self.end_date <= self.target_dates[self.current_index]:
            self.done = True

        if self.done:
            for pos in self.positions:
                self.reward += (pos * (1 if sum(self.positions) > 0 else -1)) - 1
            if self.use_cumulative_reward:
                self.reward /= max(1, len(self.positions))
            self.positions = []

        return self.state, self.reward, self.done, {"date": self.target_dates[self.current_index], "cumulative": self.cumulative_profit, "symbol": self.current_symbol}

    def reset(self):
        self.current_symbol = self.target_symbols[int(random() * len(self.target_symbols))]
        self.current_target = self.data_store[self.current_symbol]
        self.target_dates = sorted(self.current_target.keys())
        self.current_index = self.window_size
        self.positions = []
        self.cumulative_profit = 1.0
        self.done = False
        self.reward = 0
        self.update_state()
        return self.state

    def render(self, mode='human', close=False):
        if close:
            return
        return self.state

    def _seed(self):
        return int(random() * 100)

    def update_state(self):
        temp_state = []
        position_budget = (sum(self.positions) / len(self.positions)) if len(self.positions) > 0 else 1.0
        position_size = math.log(max(1.0, len(self.positions)), 100)
        position_direction = 1.0 if sum(self.positions) > 0 else 0.0
        temp_state.append([[position_budget, position_size, position_direction]])

        price_changes = []
        volume_changes = []
        for i in range(self.window_size):
            try:
                price_changes.append([self.current_target[self.target_dates[self.current_index - 1 - i]][2]])
                volume_changes.append([self.current_target[self.target_dates[self.current_index - 1 - i]][3]])
            except Exception as e:
                print(f"Error updating state: {e}")
                self.done = True
        temp_state.append([[price_changes, volume_changes]])

        temp_state = [np.array(i) for i in temp_state]
        self.state = temp_state
