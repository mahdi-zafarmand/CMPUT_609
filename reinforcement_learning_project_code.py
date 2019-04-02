import numpy as np


class Environment:
    def __init__(self):
        self.info = {}
        self.states = []
        self.actions = {}
        self.rewards = {}
        self.initialize()
        self.sigma = 0.5     
        self.P = np.zeros((len(self.states), self.actions['number']))
        self.Q = np.zeros(self.P.shape)

    def initialize(self):
        self.set_actions()
        self.reset()
        self.set_states()
        self.set_rewards()

    def set_states(self):
        size_of_environment = self.info['grid'].shape
        for i in range(size_of_environment[0] * size_of_environment[1]):
            self.states.append((i // size_of_environment[0], i % size_of_environment[1]))

    def set_actions(self):
        self.actions['RIGHT'] = 0
        self.actions['UP'] = 1
        self.actions['LEFT'] = 2
        self.actions['DOWN'] = 3
        self.actions['number'] = len(self.actions)

    def set_rewards(self):
        self.rewards['crash'] = -10
        self.rewards['checkpoint'] = 0
        self.rewards['win'] = 1000
        self.rewards['step'] = -1

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

    def reset_p_and_q(self):
        self.P = np.zeros((len(self.states), self.actions['number']))
        self.Q = np.zeros(self.P.shape)        

    def make_greedy(self, s, epsilon):
        self.P[s, :] = [epsilon / (self.actions['number'] - 1)]
        best_action = np.argmax(self.Q[s, :])
        self.P[s, best_action] = 1 - epsilon

    def choose_action(self, s, epsilon):
        self.make_greedy(s, epsilon)
        return np.random.choice(range(self.actions['number']), p=self.P[s, :])

    def choose_sigma(self, mode):
        if mode == 'SARSA':
            return 1
        elif mode == 'TreeBackUp':
            return 0
        elif mode == 'Qsigma':
            return (int) (np.random.random() < self.sigma)
        print('ERROR, incorrcet sigma mode!')
        return None

    def move(self, s, a):
        if a == 0:
            return self.move_helper(s, s + 1, '>')
        elif a == 1:
            return self.move_helper(s, s - self.info['size'], '^')
        elif a == 2:
            return self.move_helper(s, s - 1, '<')
        elif a == 3:
            return self.move_helper(s, s + self.info['size'], 'v')
        print('ERROR, incorrcet action!')
        return None

    def move_helper(self, s_prev, s_next, direction):
        self.info['grid'][self.states[s_prev]] = direction
        border = self.info['borders'][direction]
        if s_prev in border or s_next in self.info['obstacles']:
            self.reset()
            return self.info['start'], self.rewards['crash']
        self.info['grid'][self.states[s_next]] = 'O'
        if s_next == self.info['checkpoint']:
            return s_next, self.rewards['checkpoint']
        elif s_next == self.info['goal']:
            return s_next, self.rewards['win']
        else:
            return s_next, self.rewards['step']


class Episode:
    def __init__(self, environment):
        self.cum_steps = 0
        self.cum_reward = 0
        self.states_list = []
        self.actions = []
        self.p = []
        self.q = []
        self.rewards_list = []
        self.environment = environment
        self.t = -1
        self.T = np.inf
        self.tau = 0
        self.s_next = 0
        self.r = 0
        self.a_next = 0
        self.sigma = 0
    
    def initialize(self, args):
        action = self.environment.choose_action(self.environment.info['start'], args.epsilon)
        self.states_list.append(self.environment.info['start'])
        self.actions.append(action)
        self.sigmas = [1]
        self.p.append(self.environment.P[self.environment.info['start'], action])
        self.q.append(self.environment.Q[self.environment.info['start'], action])

    def update_next_state_and_reward(self):
        self.s_next, self.r = self.environment.move(self.states_list[self.t], self.actions[self.t])
        self.states_list.append(self.s_next)
        self.cum_steps += 1
        self.cum_reward += self.r

    def update_next_action_and_sigma(self, args):
        self.a_next = self.environment.choose_action(self.states_list[self.t+ 1], args.epsilon)
        self.actions.append(self.a_next)
        self.sigma = self.environment.choose_sigma(args.mode)
        self.sigmas.append(self.sigma)

    def calc_target(self, args):
        target = self.r + self.sigma * args.gamma * self.q[self.t + 1]
        target += (1 - self.sigma) * args.gamma * np.dot(self.environment.P[self.s_next, :], self.environment.Q[self.s_next, :])
        target -= self.q[self.t]
        return target

    def update_Q_and_return_tau(self, args):
        self.tau = self.t - args.n_step + 1
        if self.tau >= 0:
            E = 1
            G = self.q[self.tau]
            for k in range(self.tau, min(self.t, self.T - 1)):
                G += E * self.rewards_list[k]
                E *= args.gamma * ((1-self.sigmas[k + 1]) * self.p[k + 1] + self.sigmas[k + 1])
            error = G - self.environment.Q[self.states_list[self.tau], self.actions[self.tau]]
            self.environment.Q[self.states_list[self.tau], self.actions[self.tau]] += args.alpha * error
            self.environment.make_greedy(self.states_list[self.tau], args.epsilon)

    def run(self, episode_counter, args, n_steps, rewards):
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
                    self.q.append(self.environment.Q[self.s_next, self.a_next])
                    target = self.calc_target(args)
                    self.rewards_list.append(target)
                    self.p.append(self.environment.P[self.s_next, self.a_next])    
            self.update_Q_and_return_tau(args)
        print(self.environment.info['grid'])
        print('number of steps = ' + str(self.cum_steps))
        n_steps.append(self.cum_steps)
        rewards.append(self.cum_reward)


class Expeiment:
    def __init__(self):
        self.mode = 'TreeBackUp'
        self.n_episodes = 2000
        self.n_step = 5
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 0.05
        self.n_experiments = 10
        self.n_steps = []
        self.rewards = []
    
    def reset(self):
        self.n_steps = []
        self.rewards = []

    def run(self, environment, averages):
        environment.reset_p_and_q()
        for episode_counter in range(self.n_episodes):
            episode = Episode(environment)
            episode.run(episode_counter, self, self.n_steps, self.rewards)
        
        averages['average_steps'].append(np.average(self.n_steps))
        print('average number of steps = ' + str(averages['average_steps'][-1]))
        averages['average_reward'].append(np.average(self.rewards))
        print('average return = ' + str(averages['average_reward'][-1]))


class Project:
    def __init__(self):
        self.environment = Environment()
        self.experiment = Expeiment()
        self.averages = {}

    def initialize(self):
        self.averages['average_steps'] = []
        self.averages['average_reward'] = []

    def run(self):
        for _ in range(self.experiment.n_experiments):
            self.experiment.reset()
            self.experiment.run(self.environment, self.averages)

    def print_results(self):
        print("\nsteps: " + str(self.averages['average_steps']))
        print("steps avg: " + str(np.average(self.averages['average_steps'])))
        print("rewards: " + str(self.averages['average_reward']))
        print("rewards avg: " + str(np.average(self.averages['average_reward'])))


def main():
    project = Project()
    project.initialize()
    project.run()
    project.print_results()
    

if __name__ == "__main__":
    main()
