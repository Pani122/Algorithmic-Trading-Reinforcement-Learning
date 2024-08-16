import numpy as np
from trading_env import TradingEnvironment
from trading_model_builder import MarketModelBuilder
from keras.optimizers import SGD
import sys
import codecs

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ExperienceReplay:
    def __init__(self, max_memory=100, discount=0.9):
        self.max_memory = max_memory
        self.memory = []
        self.discount = discount

    def remember(self, states, game_over):
        """Store a transition in the replay memory."""
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def get_batch(self, model, batch_size=10):
        """Sample a batch of experiences and prepare them for training."""
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        dim = len(self.memory[0][0][0])

        inputs = [[] for _ in range(dim)]
        targets = np.zeros((min(len_memory, batch_size), num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            for j in range(dim):
                inputs[j].append(state_t[j][0])

            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])

            if game_over:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Q_sa

        inputs = [np.array(inputs[i]) for i in range(dim)]

        return inputs, targets


if __name__ == "__main__":
    code_list_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else None

    code_map = []
    with codecs.open(code_list_file, "r", "utf-8") as f:
        for line in f:
            if line.strip():
                tokens = line.strip().split(",") if "," in line else line.strip().split("\t")
                code_map.append(tokens[0])

    env = TradingEnvironment(
        data_dir="./data/",
        target_symbols=code_map,
        input_symbols=[],
        start_date="2013-08-26",
        end_date="2015-08-25"
    )

    # Parameters
    epsilon = 0.5  # Exploration rate
    min_epsilon = 0.1
    epochs = 100000
    max_memory = 5000
    batch_size = 128
    discount = 0.8

    # Build and compile the model
    model = MarketModelBuilder(model_file).get_model()
    sgd_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='rmsprop')

    # Initialize experience replay
    experience_replay = ExperienceReplay(max_memory=max_memory, discount=discount)

    # Training loop
    win_count = 0
    for epoch in range(epochs):
        loss = 0.0
        env.reset()
        game_over = False
        cum_reward = 0
        state = env.reset()

        while not game_over:
            previous_state = state
            is_random_action = False

            # Choose action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n)
                is_random_action = True
            else:
                q_values = model.predict(previous_state)
                action = np.argmax(q_values[0])

                if np.isnan(q_values).any():
                    print("Encountered NaN in Q-values!")
                    exit()

            # Take action, observe reward and next state
            state, reward, game_over, info = env.step(action)
            cum_reward += reward

            # Print action info
            if env.actions[action] in ["LONG", "SHORT"]:
                color = TerminalColors.FAIL if env.actions[action] == "LONG" else TerminalColors.OKBLUE
                if is_random_action:
                    color = TerminalColors.WARNING if env.actions[action] == "LONG" else TerminalColors.OKGREEN
                print(f"{info['dt']}:\t{color}{env.actions[action]}{TerminalColors.ENDC}\t{cum_reward:.2f}\t{info['cum']:.2f}\t" +
                      ("\t".join([f"{label}:{prob:.2f}" for label, prob in zip(env.actions, q_values[0].tolist())]) if not is_random_action else ""))

            # Store experience
            experience_replay.remember([previous_state, action, reward, state], game_over)

            # Train model on the experience batch
            inputs, targets = experience_replay.get_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)

        if cum_reward > 0 and game_over:
            win_count += 1

        print(f"Epoch {epoch:03d}/{epochs} | Loss {loss:.4f} | Win count {win_count} | Epsilon {epsilon:.4f}")

        # Save the model after each epoch
        model.save_weights(model_file or "model.h5", overwrite=True)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * 0.99)
