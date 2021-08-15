import numpy
import scipy.special
import matplotlib.pyplot
import time

# Definição de classe
class neuralNetwork:
    
    
    # Inicialização
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # Definição do treinamento
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # Definição da consulta
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# Número de nodos de entrada, saída e omissos
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# Taxa de aprendizado
learning_rate = 0.1

# Inicializa a contagem de tempo de execução
start_time = time.time()

# Instanciação da rede neural
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# Carregamento do dataset
training_data_file = open("./MNIST Dataset.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Treinamento
# "epochs" se trata do número de vezes que o treinamento é realizado por completo
epochs = 5

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# Carregamento do dataset dentro da lista
test_data_file = open("./MNIST Dataset.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Teste
# Resultado da performance, inicializado sem valor
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# Calcula a performance do algoritmo com a fração das respostas corretas
scorecard_array = numpy.asarray(scorecard)
print ("Performance: ", scorecard_array.sum() / scorecard_array.size)
print ("Time Elapsed: ", str(time.time() - start_time))