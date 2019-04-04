import numpy as np


class Environment:
    def __init__(self):
        self.info = {}
        self.states = []
        self.rewards = {}
        self.initialize()

    # this method initializes the attributes of the Environment class.
    def initialize(self):
        self.reset()
        self.set_states()
        self.set_rewards()

    # this method resets the environment, every time the agent coincides with an obstacle.
    def reset(self):
        grid = np.array([
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "G"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["X", "X", "X", "X", "X", "X", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "#", "_"],
            ["S", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
        ])
        self.info['grid'] = grid
        self.info['size'] = grid.shape[0]
        self.info['borders'] = {}
        self.info['borders']['>'] = range(grid.shape[0]-1, grid.shape[0]**2, grid.shape[0])
        self.info['borders']['<'] = range(0, grid.shape[0]**2, grid.shape[0])
        self.info['borders']['^'] = range(grid.shape[0])
        self.info['borders']['v'] = range(grid.shape[0] * (grid.shape[0] - 1), grid.shape[0]**2)
        self.info['obstacles'] = [21, 22, 23, 24, 25, 26, 27, 28, 29, 60, 61, 62, 63, 64, 65]
        self.info['start'] = 90
        self.info['goal'] = 9
        self.info['checkpoint'] = 88

    # this method defines the state number for each state.
    def set_states(self):
        size_of_environment = self.info['grid'].shape
        for i in range(size_of_environment[0] * size_of_environment[1]):
            self.states.append((i // size_of_environment[0], i % size_of_environment[1]))

    # this method determines how many reward units the agent would receive after meeting different state types.
    def set_rewards(self):
        self.rewards['crash'] = -10
        self.rewards['checkpoint'] = 0
        self.rewards['win'] = 1000
        self.rewards['step'] = -1


class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.valid_actions = {}
        self.passed_states = []
        self.n_obstacles = 0
        self.n_borders = 0
        self.sigma = 0.5
        self.set_actions()
        self.P = np.zeros((len(self.environment.states), self.valid_actions['number']))
        self.Q = np.zeros(self.P.shape)

    def reset_for_new_episode(self):
        self.passed_states.clear()
        self.n_obstacles = 0
        self.n_borders = 0
        self.sigma = 0.5

    # this method determines valid moves for the agent.
    def set_actions(self):
        self.valid_actions['RIGHT'] = 0
        self.valid_actions['UP'] = 1
        self.valid_actions['LEFT'] = 2
        self.valid_actions['DOWN'] = 3
        self.valid_actions['number'] = len(self.valid_actions)

    # this method resets the probability matrix and the state-action matrix.
    def reset_p_and_q(self):
        self.P = np.zeros((len(self.environment.states), self.valid_actions['number']))
        self.Q = np.zeros(self.P.shape)

    # this method forces the agent to act greedily w.r.t epsilon.
    def make_greedy(self, s, epsilon):
        self.P[s, :] = [epsilon / (self.valid_actions['number'] - 1)]
        best_action = np.argmax(self.Q[s, :])
        self.P[s, best_action] = 1 - epsilon

    # this method chooses the action for the agent based on the greedy behavior of the agent.
    def choose_action(self, s, epsilon):
        self.make_greedy(s, epsilon)
        return np.random.choice(range(self.valid_actions['number']), p=self.P[s, :])

    # this method chooses the sigma which is the key argument in the q-sigma algorithm, it is fixed for SARSA
    #  and TreeBackUp.
    def choose_sigma(self, alg, mode = None):
        if alg == 'SARSA':
            return 1
        elif alg == 'TreeBackUp':
            return 0
        elif alg == 'Qsigma' and mode == 'random':
            return int(np.random.random() < self.sigma)
        elif alg == 'Qsigma' and mode == 'increasing':
            self.sigma += (1 - self.sigma) / 10
            return int(np.random.random() < self.sigma)
        elif alg == 'Qsigma' and mode == 'decreasing':
            self.sigma -= (1 - self.sigma) / 10
            return int(np.random.random() < self.sigma)

    # this method moves the agent to the next state and returns the next state and the corresponding reward.
    def move(self, s, a):
        if a == 0:
            return self.move_helper(s, s + 1, '>')
        elif a == 1:
            return self.move_helper(s, s - self.environment.info['size'], '^')
        elif a == 2:
            return self.move_helper(s, s - 1, '<')
        elif a == 3:
            return self.move_helper(s, s + self.environment.info['size'], 'v')

    # this method is an  auxialary method that helps 'move' method to function.
    def move_helper(self, s_prev, s_next, direction):
        self.environment.info['grid'][self.environment.states[s_prev]] = direction
        border = self.environment.info['borders'][direction]
        if s_prev in border:
            self.n_borders += 1
            self.environment.reset()
            self.passed_states.append(self.environment.info['start'])
            return self.passed_states[-1], self.environment.rewards['crash']
        if s_next in self.environment.info['obstacles']:
            self.n_obstacles += 1
            self.environment.reset()
            self.passed_states.append(self.environment.info['start'])
            return self.passed_states[-1], self.environment.rewards['crash']
        self.environment.info['grid'][self.environment.states[s_next]] = 'O'
        if s_next == self.environment.info['checkpoint']:
            self.passed_states.append(s_next)
            return s_next, self.environment.rewards['checkpoint']
        elif s_next == self.environment.info['goal']:
            self.passed_states.append(s_next)
            return s_next, self.environment.rewards['win']
        else:
            self.passed_states.append(s_next)
            return s_next, self.environment.rewards['step']


class Episode:
    def __init__(self, environment, agent):
        self.cum_steps = 0
        self.cum_reward = 0
        self.states_list = []
        self.actions = []
        self.p = []
        self.q = []
        self.rewards_list = []
        self.environment = environment
        self.agent = agent
        self.t = -1
        self.T = np.inf
        self.tau = 0
        self.s_next = 0
        self.r = 0
        self.a_next = 0
        self.sigma = 0
        self.sigmas = []

    # this method initializes the Episode class then makes the agent choose an action.
    def initialize(self, args):
        action = self.agent.choose_action(self.environment.info['start'], args.epsilon)
        self.states_list.append(self.environment.info['start'])
        self.actions.append(action)
        self.sigmas = [1]
        self.p.append(self.agent.P[self.environment.info['start'], action])
        self.q.append(self.agent.Q[self.environment.info['start'], action])

    # this method updates the state of the agent and the received reward.
    def update_next_state_and_reward(self):
        self.s_next, self.r = self.agent.move(self.states_list[self.t], self.actions[self.t])
        self.states_list.append(self.s_next)
        self.cum_steps += 1
        self.cum_reward += self.r

    # this method updates the action of the agent and the working sigma.
    def update_next_action_and_sigma(self, args):
        self.a_next = self.agent.choose_action(self.states_list[self.t+ 1], args.epsilon)
        self.actions.append(self.a_next)
        self.sigma = self.agent.choose_sigma(args.alg, args.mode)
        self.sigmas.append(self.sigma)

    # this method calculates the target value.
    def calc_target(self, args):
        target = self.r + self.sigma * args.gamma * self.q[self.t + 1]
        target += (1 - self.sigma) * args.gamma * np.dot(self.agent.P[self.s_next, :], self.agent.Q[self.s_next, :])
        target -= self.q[self.t]
        return target

    # this method updates the q matrix and number of steps the agent should consider for the overall return value (tau)
    def update_q_and_return_tau(self, args):
        self.tau = self.t - args.n_step + 1
        if self.tau >= 0:
            e = 1
            g = self.q[self.tau]
            for k in range(self.tau, min(self.t, self.T - 1) + 1):
                g += e * self.rewards_list[k]
                e *= args.gamma * ((1-self.sigmas[k]) * self.p[k] + self.sigmas[k])
            error = g - self.agent.Q[self.states_list[self.tau], self.actions[self.tau]]
            self.agent.Q[self.states_list[self.tau], self.actions[self.tau]] += args.alpha * error
            self.agent.make_greedy(self.states_list[self.tau], args.epsilon)

    # this method makes the agent to run for a single episode.
    def run(self, args, episode_counter=0, n_steps=[], rewards=[]):
        print('\nEpisode: ' + str(episode_counter + 1) + '/' + str(args.n_episodes) + " ...")
        self.environment.reset()
        self.initialize(args)
        while self.tau != self.T - 1:
            self.t += 1
            if self.t < self.T:
                self.update_next_state_and_reward()
                if self.s_next == self.environment.info['goal']:
                    self.T = self.t + 1
                    self.rewards_list.append(self.r - self.q[self.t])
                else:
                    self.update_next_action_and_sigma(args)
                    self.q.append(self.agent.Q[self.s_next, self.a_next])
                    target = self.calc_target(args)
                    self.rewards_list.append(target)
                    self.p.append(self.agent.P[self.s_next, self.a_next])
            self.update_q_and_return_tau(args)
        print(self.environment.info['grid'])
        print('number of steps = ' + str(self.cum_steps))
        n_steps.append(self.cum_steps)
        rewards.append(self.cum_reward)


class Experiment:
    def __init__(self, n_experiments, n_episodes):
        self.alg = 'TreeBackUp'
        self.mode = 'random'
        self.n_episodes = n_episodes
        self.n_step = 5
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 0.1
        self.n_experiments = n_experiments
        self.steps = []
        self.rewards = []

    def run_a_new_experiment(self, environment, averages):
        self.steps = []
        self.rewards = []
        agent = Agent(environment)
        for episode_counter in range(self.n_episodes):
            episode = Episode(environment, agent)
            episode.run(self, episode_counter, self.steps, self.rewards)
            
        averages['average_steps'].append(np.average(self.steps))
        if len(averages['average_steps']) > 1:
            print('average number of steps = ' + str(averages['average_steps'][-1]))
        averages['average_reward'].append(np.average(self.rewards))
        print('average return = ' + str(averages['average_reward'][-1]))


class Project:
    def __init__(self, n=1, m=1):
        self.environment = Environment()
        self.agent = Agent(self.environment)
        self.experiment = Experiment(n, m)
        self.averages = {}
        self.averages['average_steps'] = []
        self.averages['average_reward'] = []

    def run(self):
        for _ in range(self.experiment.n_experiments):
            self.experiment.run_a_new_experiment(self.environment, self.averages)

    def print_results(self, detail = None):
        if detail:
            print("\nsteps: " + str(self.averages['average_steps']))
            print("steps avg: " + str(np.average(self.averages['average_steps'])))
            print("rewards: " + str(self.averages['average_reward']))
            print("rewards avg: " + str(np.average(self.averages['average_reward'])))


def main():
    # n_experiments = 1
    # n_episodes = 1
    project = Project(2, 10)
    project.run()
    project.print_results()


if __name__ == "__main__":
    main()
