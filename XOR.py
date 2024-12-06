import math
import random
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import csv


city_scale = 15 
padding = 100


class Neuron:
    def __init__(self, layer_index, neuron_index, x, y):
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.x = x
        self.y = y
        self.output = 0.0
        self.delta = 0.0
        self.bias = random.uniform(-1, 1)

    def draw(self, canvas):
        canvas.create_oval(self.x - city_scale, self.y - city_scale, self.x + city_scale, self.y + city_scale,
                           fill='white', outline='black')
        canvas.create_text(self.x, self.y, text=f"{self.output:.2f}", font=("Arial", 8))


class Weight:
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.value = random.uniform(-1, 1)

    def draw(self, canvas, color='gray'):
        canvas.create_line(self.from_neuron.x, self.from_neuron.y, self.to_neuron.x, self.to_neuron.y, fill=color)
        mid_x = (self.from_neuron.x + self.to_neuron.x) / 2
        mid_y = (self.from_neuron.y + self.to_neuron.y) / 2
        canvas.create_text(mid_x, mid_y, text=f"{self.value:.2f}", font=("Arial", 8))


class DataPreprocessor:
    @staticmethod
    def load_csv(filepath):
        data = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                try:
                    inputs = [float(x) for x in row[:-1]]  
                    target = [float(row[-1])]  
                    data.append((inputs, target))
                except ValueError as e:
                    print(f"Skipping invalid row {row}: {e}")
        return data




class NeuralNetwork:
    def __init__(self, layers, ui, activation_function):
        self.layers = layers
        self.neurons = []
        self.weights = []
        self.ui = ui
        self.create_network()
        self.activation_function = activation_function

    def create_network(self):
        num_layers = len(self.layers)
        layer_width = (self.ui.w - 2 * padding) / (num_layers - 1 if num_layers > 1 else 1)
        self.neurons = []
        for l_index, num_neurons in enumerate(self.layers):
            layer_neurons = []
            layer_height = (self.ui.h - 2 * padding) / (num_neurons - 1 if num_neurons > 1 else 1)
            for n_index in range(num_neurons):
                x = padding + l_index * layer_width
                y = padding + n_index * layer_height
                neuron = Neuron(l_index, n_index, x, y)
                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)

        self.weights = []
        for l in range(len(self.neurons) - 1):
            for from_neuron in self.neurons[l]:
                for to_neuron in self.neurons[l + 1]:
                    weight = Weight(from_neuron, to_neuron)
                    self.weights.append(weight)

    def forward(self, inputs):
        for i, value in enumerate(inputs):
            self.neurons[0][i].output = value

        for l in range(1, len(self.layers)):
            for neuron in self.neurons[l]:
                total_input = neuron.bias
                for prev_neuron in self.neurons[l - 1]:
                    for weight in self.weights:
                        if weight.from_neuron == prev_neuron and weight.to_neuron == neuron:
                            total_input += prev_neuron.output * weight.value
                            self.ui.draw_forward_connection(weight)
                            self.ui.update()
                            self.ui.after(50)
                neuron.output = self.activation(total_input)
                neuron.draw(self.ui.canvas)
                self.ui.update()
                self.ui.after(50)

        outputs = [neuron.output for neuron in self.neurons[-1]]
        return outputs

    def activation(self, x):
        if self.activation_function == 'Sigmoid':
            return 1 / (1 + math.exp(-x))
        elif self.activation_function == 'Tanh':
            return math.tanh(x)
        elif self.activation_function == 'ReLU':
            return max(0, x)

    def activation_derivative(self, output):
        if self.activation_function == 'Sigmoid':
            return output * (1 - output)
        elif self.activation_function == 'Tanh':
            return 1 - output ** 2
        elif self.activation_function == 'ReLU':
            return 1 if output > 0 else 0

    def backward(self, targets, learning_rate):
        # Validate the target length matches the output layer
        if len(targets) != len(self.neurons[-1]):
            raise ValueError(f"Target length {len(targets)} does not match output layer size {len(self.neurons[-1])}")

        # Compute deltas for output layer
        for i, neuron in enumerate(self.neurons[-1]):
            error = targets[i] - neuron.output
            neuron.delta = error * self.activation_derivative(neuron.output)

        # Compute deltas for hidden layers
        for l in reversed(range(1, len(self.layers) - 1)):
            for neuron in self.neurons[l]:
                error = 0.0
                for weight in self.weights:
                    if weight.from_neuron == neuron:
                        error += weight.to_neuron.delta * weight.value
                neuron.delta = error * self.activation_derivative(neuron.output)

        # Update weights
        for weight in self.weights:
            change = learning_rate * weight.from_neuron.output * weight.to_neuron.delta
            weight.value += change
            self.ui.draw_backward_connection(weight)
            self.ui.update()
            self.ui.after(10)

        # Update biases
        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.bias += learning_rate * neuron.delta


class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Neural Network Visualization")
        self.option_add("*tearOff", FALSE)
        width, height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (width, height))
        self.state("zoomed")
        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=width, height=height)
        self.w = width - padding
        self.h = height - padding * 2
        self.network = None
        self.training_data = None
        self.testing_data = None
        self.create_menu()

    def create_menu(self):
        menu_bar = Menu(self)
        self['menu'] = menu_bar
        menu_NN = Menu(menu_bar)
        menu_bar.add_cascade(menu=menu_NN, label='Neural Network', underline=0)
        menu_NN.add_command(label="Generate Network", command=self.generate_network, underline=0)
        menu_NN.add_command(label="Start Training", command=self.start_training, underline=0)
        menu_NN.add_command(label="Start Testing", command=self.start_testing, underline=0)

        control_frame = Frame(self)
        control_frame.pack(side='left', padx=10)

        self.layer_label = Label(control_frame, text="Layers (comma-separated):")
        self.layer_label.pack()
        self.layer_entry = Entry(control_frame)
        self.layer_entry.pack()
        self.layer_entry.insert(0, '2,2,1')

        self.lr_label = Label(control_frame, text="Learning Rate:")
        self.lr_label.pack()
        self.lr_entry = Entry(control_frame)
        self.lr_entry.pack()
        self.lr_entry.insert(0, '0.5')

        # New field for number of epochs
        self.epoch_label = Label(control_frame, text="Number of Epochs:")
        self.epoch_label.pack()
        self.epoch_entry = Entry(control_frame)
        self.epoch_entry.pack()
        self.epoch_entry.insert(0, '100')  # Default to 100 epochs

        self.activation_label = Label(control_frame, text="Activation Function:")
        self.activation_label.pack()
        self.activation_var = StringVar(value="Sigmoid")
        for func in ['Sigmoid', 'Tanh', 'ReLU']:
            Radiobutton(control_frame, text=func, variable=self.activation_var, value=func).pack(anchor=W)

        self.load_data_button = Button(control_frame, text="Load CSV", command=self.load_csv)
        self.load_data_button.pack()

    def generate_network(self):
        layers = [int(s) for s in self.layer_entry.get().split(',')]
        activation_function = self.activation_var.get()
        self.network = NeuralNetwork(layers, self, activation_function)
        self.draw_network(self.network)

    def draw_network(self, network):
        self.canvas.delete("all")
        for weight in network.weights:
            weight.draw(self.canvas)
        for layer in network.neurons:
            for neuron in layer:
                neuron.draw(self.canvas)

    def draw_forward_connection(self, weight):
        weight.draw(self.canvas, color='blue')
        self.canvas.update()

    def draw_backward_connection(self, weight):
        weight.draw(self.canvas, color='red')
        self.canvas.update()

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            data = DataPreprocessor.load_csv(filepath)
            split_idx = int(0.8 * len(data)) 
            self.training_data = data[:split_idx]
            self.testing_data = data[split_idx:]
            print(f"Loaded {len(self.training_data)} training samples and {len(self.testing_data)} testing samples.")

    def start_training(self):
        if self.network and self.training_data:
            learning_rate = float(self.lr_entry.get())
            total_epochs = int(self.epoch_entry.get())

            print("Length of training_data:", len(self.training_data))
            print(f"Starting training for {total_epochs} epochs with {len(self.training_data)} training samples...")
            print("--------------------------------------------------------")

            for epoch in range(total_epochs):
                epoch_loss = 0.0
                for inputs, targets in self.training_data:
                    outputs = self.network.forward(inputs)
                    self.network.backward(targets, learning_rate)
                    sample_loss = sum((t - o) ** 2 for t, o in zip(targets, outputs)) / len(targets)
                    epoch_loss += sample_loss

                epoch_loss /= len(self.training_data)
                print(f"Epoch {epoch + 1}/{total_epochs}: Loss = {epoch_loss:.4f}")

            print("--------------------------------------------------------")
            print("Training complete!")

            # Confirm we have data before accuracy calculation
            if len(self.training_data) == 0:
                print("No training data available to calculate accuracy.")
                return

            print("Calculating final training accuracy...")
            correct = 0
            for inputs, targets in self.training_data:
                outputs = self.network.forward(inputs)
                # Print debug info:
                # print("Acc Check - Inputs:", inputs, "Targets:", targets, "Outputs:", outputs)
                if int(round(outputs[0])) == int(targets[0]):
                    correct += 1

            training_accuracy = (correct / len(self.training_data)) * 100
            print(f"Final Training Accuracy: {training_accuracy:.2f}%")

    def start_testing(self):
        if self.network and self.testing_data:
            print(f"Starting testing with {len(self.testing_data)} testing samples...")
            correct = 0
            total_loss = 0.0
            for i, (inputs, targets) in enumerate(self.testing_data, start=1):
                outputs = self.network.forward(inputs)
                sample_loss = sum((t - o) ** 2 for t, o in zip(targets, outputs)) / len(targets)
                total_loss += sample_loss

                prediction_correct = int(round(outputs[0]) == targets[0])
                correct += prediction_correct
                print(
                    f"Test Sample {i}: Inputs={inputs}, Targets={targets}, Outputs={outputs}, Correct={prediction_correct}")

            accuracy = (correct / len(self.testing_data)) * 100
            avg_loss = total_loss / len(self.testing_data)
            print("--------------------------------------------------------")
            print(f"Testing complete. Final Testing Accuracy: {accuracy:.2f}% | Average Loss: {avg_loss:.4f}")
            print("--------------------------------------------------------")


if __name__ == '__main__':
    ui = UI()
    ui.mainloop()
    