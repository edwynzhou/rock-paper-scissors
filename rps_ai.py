import numpy as np  
import copy
import random

# Counter of how many times rock, paper, or scissors are chosen
global_rps = [0, 0, 0]

# Activation Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Alter value of nodes for iteration
def mutate(x):
    return x + ((random.random()-0.5)*2)*0.1

# Neural network 
class network:
    def __init__(self, layer_sizes):
        self.weights  = []
        self.ls = layer_sizes
        for l in range(0, len(layer_sizes)-1):
            new_layer = np.random.normal(size=(layer_sizes[l+1], layer_sizes[l]))
            self.weights.append(new_layer)

    def feedforward(self, input_vector):
        current = input_vector
        sig = np.vectorize(sigmoid)
        for w in self.weights:
            current = sig(w*current)
        return current
    
    def iterate(self):
        child = network(self.ls)
        child.weights = copy.deepcopy(self.weights)
        mod = np.vectorize(mutate)
        for w in range(len(child.weights)):
            child.weights[w] = mod(child.weights[w])
        return child

# Create network
agents = []
for i in range(100):
    agents.append(network([3, 3, 3]))

# Fitness function    
def get_fitness(ag):
    # r = 0, p = 1, s = 2
    # provide training data of rock option
    rps_alg = [0,0,0,0,0,0]
    fitness = 0
    rps = [0, 0, 0]
    # Compare agent output to desired output
    for i in range(len(rps_alg)):
        out: np.matrix = ag.feedforward(np.transpose(np.matrix(rps)))
        if rps_alg[i] == 0:
            answer = 1
        elif rps_alg[i] == 1:
            answer = 2
        elif rps_alg[i] == 2:
            answer = 0
        global_rps[rps_alg[i]] += 1
        answer_arr = [-1, -1, -1]
        answer_arr[answer] = 1
        # Fitness function
        fitness = np.dot(np.transpose(out).tolist(), answer_arr).tolist()[0]
    return float(fitness) / float(len(rps_alg))

def check_fitness(ag, choice):
    # r = 0, p = 1, s = 2
    rps_alg = []
    rps_alg += [0,0,0,0,0,0]
    fitness = 0
    out = ag.feedforward(np.transpose(np.matrix(global_rps)))
    global_rps[choice] += 1
    print(out)
    return out

# Run iterations of network to copy over the best performing bot 
for i in range(100):
    best: network = None
    best_fitness = -9999
    for agent in agents:
        fitness = get_fitness(agent)
        test_count = 10
        for t in range(test_count):
            fitness += get_fitness(agent)
        fitness /= test_count
        if fitness > best_fitness:
            best_fitness = fitness
            best = agent
    # Output choice of rock [1,0,0]
    # paper = [0,1,0], scissors = [0,0,1]
    print("Best Fitness       Rock         Paper       Scissors")
    print(best_fitness, np.transpose(best.feedforward(np.transpose(np.matrix([1,0,0])))))
    agents = []
    for x in range(100):
        agents.append(best.iterate())
    print("Iteration " + str(i))