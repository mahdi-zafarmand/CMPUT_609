import numpy as np


class World:
    def __init__(self):
        self.environment = {}
        self.states = []
        self.velocity = {}
        self.actions = {}
        self.rewards = {}
        self.initialize()
        self.sigma = 0.5     
        self.checkpoint = False   
        self.P = np.zeros((len(self.states), len(self.actions['actions'])))
        self.Q = np.zeros(self.P.shape)

    def initialize(self):
        self.set_actions()
        self.reset()
        self.set_states()
        self.set_rewards()

    def set_states(self):
        size_of_environment = self.environment['grid'].shape
        for i in range(size_of_environment[0] * size_of_environment[1]):
            self.states.append((i // size_of_environment[0], i % size_of_environment[1]))

    def set_actions(self):
        self.actions['RIGHT'] = 0
        self.actions['UP'] = 1
        self.actions['LEFT'] = 2
        self.actions['DOWN'] = 3
        self.actions['actions'] = range(len(self.actions))

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
        self.environment['grid'] = grid
        self.environment['size'] = grid.shape[0]
        self.environment['borders'] = {}
        self.environment['borders']['>'] = range(grid.shape[0]-1, grid.shape[0]**2, grid.shape[0])
        self.environment['borders']['<'] = range(0, grid.shape[0]**2, grid.shape[0])
        self.environment['borders']['^'] = range(grid.shape[0])
        self.environment['borders']['v'] = range(grid.shape[0] * (grid.shape[0] - 1), grid.shape[0]**2)
        self.environment['obstacles'] = [21, 22, 23, 24, 25, 26, 27, 28, 29, 60, 61, 62, 63, 64, 65]
        self.environment['start'] = 90
        self.environment['goal'] = 9
        self.environment['checkpoint'] = 88

    def reset_p_and_q(self):
        self.P = np.zeros((len(self.states), len(self.actions['actions'])))
        self.Q = np.zeros(self.P.shape)        

    def make_greedy(self, s, epsilon):
        ACTIONS = self.actions['actions']
        self.P[s, :] = [epsilon / (len(ACTIONS) - 1)]
        best_a = np.argmax(self.Q[s, :])
        self.P[s, best_a] = 1 - epsilon

    def choose_action(self, s, epsilon):
        self.make_greedy(s, epsilon)
        return np.random.choice(self.actions['actions'], p=self.P[s, :])

    def choose_sigma(self, mode):
        if mode == 'SARSA':
            return 1
        elif mode == 'TreeBackUp':
            return 0
        elif mode == 'Qsigma':
            return (int) (np.random.random() < self.sigma)
        else:
            print('ERROR, incorrcet sigma mode!')
            return None

    def move(self, s, a):
        self.checkpoint = False
        if a == 0:
            return self.move_helper(s, s + 1, '>')
        elif a == 1:
            return self.move_helper(s, s - self.environment['size'], '^')
        elif a == 2:
            return self.move_helper(s, s - 1, '<')
        elif a == 3:
            return self.move_helper(s, s + self.environment['size'], 'v')
        else:
            print('ERROR, incorrcet action!')
            return None

    def move_helper(self, s_prev, s_next, direction):
        self.environment['grid'][self.states[s_prev]] = direction
        border = self.environment['borders'][direction]
        if s_prev in border or s_next in self.environment['obstacles']:
            self.reset()
            return self.environment['start'], self.rewards['crash']
        self.environment['grid'][self.states[s_next]] = 'O'
        if s_next == self.environment['checkpoint']:
            self.checkpoint = True
            return s_next, self.rewards['checkpoint']
        elif s_next == self.environment['goal']:
            return s_next, self.rewards['win']
        else:
            return s_next, self.rewards['step']


class Experiment:
    def __init__(self):
        self.steps = 0
        self.reward = 0
        self.states = []
        self.actions = []
        self.p = []
        self.q = []
        self.targets = []


def set_args():
    args = dict()
    args['mode'] = 'TreeBackUp'
    args['n_episodes'] = 5000
    args['n_steps'] = 5
    args['gamma'] = 0.99
    args['alpha'] = 0.1
    args['epsilon'] = 0.1
    args['n_experiments'] = 10
    return args
    

def setup_experiment():
    n_steps = []
    rewards = []
    return n_steps, rewards


def initialize_episode_terms():
    episode_terms = {}
    episode_terms['steps'] = 0
    episode_terms['reward'] = 0
    episode_terms['states'] = []
    episode_terms['actions'] = []
    episode_terms['p'] = []
    episode_terms['q'] = []
    episode_terms['targets'] = []
    return episode_terms


def initialize_episode(world, args):
    episode_terms = initialize_episode_terms()
    action = world.choose_action(world.environment['start'], args['epsilon'])
    episode_terms['states'].append(world.environment['start'])
    episode_terms['actions'].append(action)
    episode_terms['sigmas'] = [1]
    episode_terms['p'].append(world.P[world.environment['start'], action])
    episode_terms['q'].append(world.Q[world.environment['start'], action])
    return -1, 0, np.inf, episode_terms


def update_Q_and_return_tau(world, args, episode_terms, t, T):
    tau = t - args['n_steps'] + 1
    if tau >= 0:
        E = 1
        G = episode_terms['q'][tau]
        for k in range(tau, min(t, T - 1)):
            G += E * episode_terms['targets'][k]
            E *= args['gamma'] * ((1-episode_terms['sigmas'][k + 1]) * episode_terms['p'][k + 1] + episode_terms['sigmas'][k + 1])
        error = G - world.Q[episode_terms['states'][tau], episode_terms['actions'][tau]]
        world.Q[episode_terms['states'][tau], episode_terms['actions'][tau]] += args['alpha'] * error
        world.make_greedy(episode_terms['states'][tau], args['epsilon'])
    return tau


def get_next_state_and_reward(world_instance, episode_terms, t):
    s_next, r = world_instance.move(episode_terms['states'][t], episode_terms['actions'][t])
    episode_terms['states'].append(s_next)
    episode_terms['steps'] += 1
    episode_terms['reward'] += r
    return s_next, r


def calc_target(r, sigma, args, episode_terms, t, world_instance, s_next):
    target = r + sigma * args['gamma'] * episode_terms['q'][t + 1]
    target += (1 - sigma) * args['gamma'] * np.dot(world_instance.P[s_next, :], world_instance.Q[s_next, :])
    target -= episode_terms['q'][t]
    return target


def get_next_action_and_sigma(world_instance, episode_terms, t, args):
    a_next = world_instance.choose_action(episode_terms['states'][t+ 1], args['epsilon'])
    episode_terms['actions'].append(a_next)
    sigma = world_instance.choose_sigma(args['mode'])
    episode_terms['sigmas'].append(sigma)
    return a_next, sigma


def run_an_episode(world_instance, episode, args, n_steps, rewards):
    print('\nEpisode: ' + str(episode + 1) + '/' + str(args['n_episodes']) + " ...")
    world_instance.reset()
    t, tau, T, episode_terms = initialize_episode(world_instance, args)
    while tau != T - 1:
        t += 1
        if t < T:
            s_next, r = get_next_state_and_reward(world_instance, episode_terms, t)
            if s_next == world_instance.environment['goal']:
                T = t + 1
                episode_terms['targets'].append(r - episode_terms['q'][t])
            else:
                a_next, sigma = get_next_action_and_sigma(world_instance, episode_terms, t, args)
                episode_terms['q'].append(world_instance.Q[s_next, a_next])
                target = calc_target(r, sigma, args, episode_terms, t, world_instance, s_next)
                episode_terms['targets'].append(target)
                episode_terms['p'].append(world_instance.P[s_next, a_next])
        tau = update_Q_and_return_tau(world_instance, args, episode_terms, t, T)
    
    print(world_instance.environment['grid'])
    print('number of steps = ' + str(episode_terms['steps']))
    n_steps.append(episode_terms['steps'])
    rewards.append(episode_terms['reward'])

def run_an_experiment(args, world_instance, averages):
    n_steps, rewards = setup_experiment()
    # world_instance.reset_p_and_q()
    for episode in range(args['n_episodes']):
        run_an_episode(world_instance, episode, args, n_steps, rewards)
    
    averages['average_steps'].append(np.average(n_steps))
    print('average number of steps = ' + str(averages['average_steps'][-1]))
    averages['average_reward'].append(np.average(rewards))
    print('average return = ' + str(averages['average_reward'][-1]))

def main():
    world_instance = World()
    args = set_args()
    averages = {}
    averages['average_steps'] = []
    averages['average_reward'] = []
    for _ in range(args['n_experiments']):
        run_an_experiment(args, world_instance, averages)

    print("\nsteps: " + str(averages['average_steps']))
    print("steps avg: " + str(np.average(averages['average_steps'])))
    print("rewards: " + str(averages['average_reward']))
    print("rewards avg: " + str(np.average(averages['average_reward'])))


if __name__ == "__main__":
    main()
