import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt


class Details:

        dimensions = 4
        length = 20
        data_file = "drinks_examples.csv"
        file_text = []
        tol = 10**(-8)


class Network:
    
    def __init__(self):
        self.iteration = 0
        self.neuron_data = np.empty([Details.length, Details.length], dtype=Neuron)
        self.input_data = []
        self.input_datum = []

    def activate(self):
        for i in range(0,Details.length):
            for j in range(0,Details.length):
                self.neuron_data[j][i] = Neuron(i, j)
                for f in range(0,Details.dimensions):
                    self.neuron_data[j][i].weights[f] = (i+j+f + 1)/(i**2 + j**2 + f**2 + i+j+f + 2)

    def load_data(self):
        with open(Details.data_file) as input_file:
            ignore_first = 0
            for line in input_file:
                if ignore_first != 0:
                    lines = line.replace("\out", "").split(",")
                    self.input_data.append(lines[1:])
                    Details.file_text.append(lines[0])
                ignore_first = ignore_first + 1
        for x in range(0, Details.dimensions):
            all_sum = 0
            for y in range(0, len(self.input_data)):
                all_sum += float(self.input_data[y][x])
            average = all_sum/(len(self.input_data))
            for y in range(0, len(self.input_data)):
                self.input_data[y][x] = float(self.input_data[y][x])/average

    def train(self):
        current_error = 100000000.0
        selected_datum = []
        counter = 0
        while current_error > Details.tol:
            current_error = 0
            for x in range(0, len(self.input_data)):
                selected_datum.append(self.input_data[x])
            for y in range(0,len(self.input_data)):
                self.input_datum = np.asarray(selected_datum[rnd.randint(0, (len(self.input_data) - 1 - y))])
                current_error += Network.train_data()
                selected_datum.remove(self.input_datum.tolist())
            counter = counter + 1
            print("Iteration: " + str(counter) + "| Training Error: " + str(current_error))

    def train_data(self):
        error = 0
        optimal_neuron = Network.top_selection()
        for x in range(0, Details.length):
            for y in range(0, Details.length):
                error += self.neuron_data[x][y].update_weights(self.input_datum, optimal_neuron, self.iteration)
        self.iteration = self.iteration + 1
        return abs(error/(Details.length**2))

    @staticmethod
    def euclidean(vector1, vector2):
        distance = 0
        for i in range(0, len(vector1)):
            distance += (vector1[i] - vector2[i])**2
        return distance**(1/2)

    def top_selection(self):
        minimum = 100000000
        optimal_neuron = Neuron
        for x in range(0, Details.length):
            for y in range(0, Details.length):
                distance = Network.euclidean(self.input_datum, self.neuron_data[x][y].weights)
                if distance < minimum:
                    minimum = distance
                    optimal_neuron = self.neuron_data[x][y]
        return optimal_neuron

    def display_results(self):
        xs = []
        ys = []
        for i in range(0, len(self.input_data)):
            self.input_datum = self.input_data[i]
            optimal = Network.top_selection()
            print(Details.file_text[i] + " | X=" + str(optimal.X) + "Y=" + str(optimal.Y))
            xs.append(optimal.X)
            ys.append(optimal.Y)
        plt.scatter(xs, ys)
        labels = [Details.file_text[i] for i in range(len(Details.file_text))]
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-5, 5),
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='red', alpha=0.05))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Self-Organising Network Mapping")
        plt.xlim(-0.5, (Details.length + 0.5))
        plt.ylim(-0.5, (Details.length + 0.5))
        plt.grid()
        plt.show()

    def create_net(self):
        self.activate()
        self.load_data()
        self.train()
        self.display_results()


class Neuron:

    def __init__(self, x, y):
        self.weights = np.zeros(Details.dimensions)
        self.X = x
        self.Y = y
        self.length = Details.length
        self.factor = 1000/np.log(Details.length)

    def update_weights(self, input_datum, optimal_neuron, count):
        summed = 0
        for i in range(0, len(self.weights)):
            delta = math.exp(-count/1000)*0.1 * Neuron.gaussian(self, optimal_neuron, count)\
                  * (input_datum[i] - self.weights[i])
            self.weights[i] += delta
            summed += delta
        return summed/len(self.weights)

    def gaussian(self, optimal_neuron, c):
        euclidean = ((optimal_neuron.X - self.X)**2 + (optimal_neuron.Y - self.Y)**2) ** 1/2
        return math.exp(-(euclidean**2)/(Neuron.learn_weight(self, c))**2)

    def learn_weight(self, c):
        return math.exp(-c/self.factor)*self.length


Network = Network()
Network.create_net()
