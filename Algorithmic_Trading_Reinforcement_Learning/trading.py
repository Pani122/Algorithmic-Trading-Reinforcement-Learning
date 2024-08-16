import os
import numpy as np
from keras.optimizers import SGD
from trading_env import TradingEnvironment
from trading_model_builder import MarketPolicyGradientModelBuilder


class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PolicyGradientAgent:

    def __init__(self, environment, discount_rate=0.99, model_file=None, history_file=None):
        self.env = environment
        self.discount_rate = discount_rate
        self.model_file = model_file
        self.history_file = history_file

        self.model = MarketPolicyGradientModelBuilder(self.model_file).get_model()
        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer='rmsprop')

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_sum = 0
        rewards = rewards.flatten()

        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_sum = 0

            running_sum = running_sum * self.discount_rate + rewards[t]
            discounted_rewards[t] = running_sum

        return discounted_rewards

    def train(self, max_episodes=1000000, max_steps_per_episode=200, verbosity=0):
        env = self.env
        model = self.model
        avg_reward_sum = 0.0

        for episode in range(max_episodes):
            env.reset()
            observation = env.reset()
            game_over = False
            reward_sum = 0

            observations = []
            actions = []
            predicted_probs = []
            rewards = []

            while not game_over:
                action_probabilities = model.predict(observation)[0]
                observations.append(observation)
                predicted_probs.append(action_probabilities)

                if action_probabilities.shape[0] > 1:
                    action = np.random.choice(self.env.action_space.n, 1, p=action_probabilities / np.sum(action_probabilities))[0]

                    one_hot_action = np.zeros([self.env.action_space.n])
                    one_hot_action[action] = 1.0
                    actions.append(one_hot_action)
                else:
                    action = 0 if np.random.uniform() < action_probabilities else 1
                    actions.append([float(action)])

                observation, reward, game_over, info = self.env.step(action)
                reward_sum += float(reward)
                rewards.append(float(reward))

                if verbosity > 0:
                    if env.actions[action] in ["LONG", "SHORT"]:
                        color = TerminalColors.FAIL if env.actions[action] == "LONG" else TerminalColors.OKBLUE
                        print(f"{info['dt']}:\t{color}{env.actions[action]}{TerminalColors.ENDC}\t{reward_sum:.2f}\t{info['cum']:.2f}\t" +
                              "\t".join([f"{label}:{prob:.2f}" for label, prob in zip(env.actions, action_probabilities.tolist())]))

            avg_reward_sum = avg_reward_sum * 0.99 + reward_sum * 0.01
            output_string = f"{episode}\t{info['code']}\t{(TerminalColors.FAIL if reward_sum >= 0 else TerminalColors.OKBLUE)}" + \
                            f"{reward_sum:.2f}{TerminalColors.ENDC}\t{info['cum']:.2f}\t{avg_reward_sum:.2f}"
            print(output_string)

            if self.history_file:
                with open(self.history_file, 'a') as history:
                    history.write(f"{output_string}\n")

            observations_reshaped = np.array([np.array(obs).flatten() for obs in observations])
            actions_reshaped = np.vstack(actions)
            predicted_probs_reshaped = np.vstack(predicted_probs)
            rewards_reshaped = np.vstack(rewards)

            discounted_rewards = self.discount_rewards(rewards_reshaped)
            discounted_rewards /= np.std(discounted_rewards)

            for i, (reward, discounted_reward) in enumerate(zip(rewards, discounted_rewards)):
                if verbosity > 1:
                    print(actions_reshaped[i], end=' ')

                if discounted_reward < 0:
                    actions_reshaped[i] = 1 - actions_reshaped[i]
                    actions_reshaped[i] /= sum(actions_reshaped[i])

                actions_reshaped[i] = np.clip(predicted_probs_reshaped[i] + (actions_reshaped[i] - predicted_probs_reshaped[i]) * abs(discounted_reward), 0, 1)

                if verbosity > 1:
                    print(predicted_probs_reshaped[i], actions_reshaped[i], reward, discounted_reward)

            model.fit(observations_reshaped, actions_reshaped, epochs=1, verbose=0, shuffle=True)
            model.save_weights(self.model_file)


if __name__ == "__main__":
    import sys
    import codecs

    code_list_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else None
    history_file = sys.argv[3] if len(sys.argv) > 3 else None

    code_map = {}
    with codecs.open(code_list_file, "r", "utf-8") as f:
        for line in f:
            if line.strip():
                tokens = line.strip().split(",") if "," in line else line.strip().split("\t")
                code_map[tokens[0]] = tokens[1]

    env = TradingEnvironment(dir_path="./data/", target_codes=list(code_map.keys()), input_codes=[], start_date="2010-08-25", end_date="2015-08-25", sudden_death=-1.0)

    policy_gradient_agent = PolicyGradientAgent(env, discount_rate=0.9, model_file=model_file, history_file=history_file)
    policy_gradient_agent.train(verbosity=1)
