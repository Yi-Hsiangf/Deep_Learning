import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=1000):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval

        # Model parameters initialization
        # Please initiate your network parameters here.
        self.w1 = np.array([[random.random() for i in range(hidden_size)] for j in range(2)]) #2 * hidden_size matrix
        self.w2 = np.array([[random.random() for i in range(hidden_size)] for j in range(hidden_size)])
        self.w3 = np.array([[random.random()] for j in range(hidden_size)])
        
        
        self.learning_rate = 0.05
        
        self.A1 = []
        self.A2 = []
        self.Y = []
        
        ...

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        """
        #print("w1 shape", self.w1.shape)
        z1 = np.dot(inputs, self.w1)
        self.a1 = sigmoid(z1)
        
        z2 = np.dot(self.a1, self.w2)
        self.a2 = sigmoid(z2)
        
        z3 = np.dot(self.a2, self.w3)
        self.y = sigmoid(z3)
                
        return self.y

    def backward(self, i):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """
        
        #Compute gradient for w1, w2, w3
        w1_grad = np.zeros((2, 3))
        w2_grad = np.zeros((3, 3))
        w3_grad = np.zeros((3, 1))
        
        
        w3_backward_pass = np.zeros((1, 1))
        w2_backward_pass = np.zeros((1, 3))
    
        #print("self.error shape",self.error.shape)
        #Compute w3 gradient
        for i, w in enumerate(w3_grad):      # 3 x 1 
            w3_forward_pass = self.a2[0][i]
            w3_backward_pass = self.error * der_sigmoid(self.y)
            w3_grad[i] = w3_forward_pass * w3_backward_pass
        
        #Compute w2 gradient
        for i, w_row in enumerate(w2_grad):   # 3 x 3 
            for j, w in enumerate(w2_grad[i]):# 1 x 3 
                w2_forward_pass = self.a1[0][i]
                w2_backward_pass[0][i] =  der_sigmoid(self.a2[0][i]) * self.w3[i][0] * w3_backward_pass
                w2_grad[i][j] = w2_forward_pass * w2_backward_pass[0][i]
        
        
        #Compute w1 gradient 
        for i, w_rol in enumerate(w1_grad):    # 2 x 3
            for j, w in enumerate(w1_grad[i]): # 1 x 3
                w1_forward_pass = self.input[0][i]
                w1_backward_pass = der_sigmoid(self.a1[0][i]) * self.w2[i][j] * w2_backward_pass[0][i]
                w1_grad[i][j] = w1_forward_pass * w1_backward_pass
        
        
        #Update 
        for i, w in enumerate(w3_grad): 
            self.w3[i] -= self.learning_rate * w3_grad[i]
            
        for i, w_row in enumerate(w2_grad):   # 3 x 3 
            for j, w in enumerate(w2_grad[i]):# 1 x 3 
                self.w2[i][j] -= self.learning_rate * w2_grad[i][j]
                
        for i, w_rol in enumerate(w1_grad):    # 2 x 3
            for j, w in enumerate(w1_grad[i]): # 1 x 3
                self.w1[i][j] -= self.learning_rate * w1_grad[i][j]
        
        #print("w3 grad : ", w3_grad)
        #print("w3.shape :", self.w3.shape)
        
        
    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]
        self.X = inputs
        for epochs in range(self.num_step):
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.input = inputs[idx:idx+1, :]
                self.output = self.forward(inputs[idx:idx+1, :])
                self.error = self.output - labels[idx:idx+1, :] #derevative of cross entropy
                self.backward(idx)

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

        print('Training finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]

        error = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])

        print("error: ", error)
        error /= n
        print('accuracy: %.2f' % ((1 - error)*100) + '%')
        print('')


if __name__ == '__main__':
    #data, label = GenData.fetch_data('Linear', 5)
    data = np.array([[0.13772156, 0.13847214],[0.37425389, 0.88789611], [0.03151926, 0.78380487], [0.51589588, 0.10269717], [0.62426485, 0.67746004]])
    label = np.array([[1], [1], [1], [0], [1]])
    print("data shape: ", data.shape)
    print("label shape", label.shape)
    net = SimpleNet(3, num_step=1000000)
    net.train(data, label)
    
    pred_result = np.round(net.forward(data))
    SimpleNet.plot_result(data, label, pred_result)

