from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import torch
import math
import scipy.io as sio

import Silence_combined

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.fc4 = nn.Linear(10, 3)
        # self.fc5 = nn.Linear(10, 3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = F.log_softmax(x, dim=1)

        return x

class NNModel:
    def __init__(self, network, learning_rate, data):

        self.model = network

        # self.trainloader = torch.utils.data.DataLoader(data[0], batch_size=10, shuffle=True)
        # self.testloader = torch.utils.data.DataLoader(data[1], batch_size=10, shuffle=True)

        self.trainloader = data[0]
        self.testloader = data[1]

        self.lossfn = F.nll_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_train_samples = len(self.trainloader)
        self.num_test_samples = len(self.testloader)

    def train_epoch(self):
        self.model = self.model.float()
        self.model.train()

        loss = None

        for images, labels in self.trainloader:

            # labels_list = [labels]
            labels = torch.tensor(labels, dtype=torch.long)
            #labels = labels.clone(detype=torch.long).detach()

            log_ps = self.model(images.float())

            self.optimizer.zero_grad()

            loss = self.lossfn(log_ps, labels)

            loss.backward()
            self.optimizer.step()

        print(loss)

         # save the trained model.
        torch.save(self.model, './model.pth')

        return

    def eval(self):
        self.model = self.model.float()
        self.model.eval()
        #accuracy = 0

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.testloader:

                # labels_list = [labels]
                labels = torch.tensor(labels, dtype=torch.long)
                #labels = labels.clone(detype=torch.long).detach()

                log_ps = self.model(images.float())

                for idx, i in enumerate(log_ps):
                    if torch.argmax(i) == labels[idx]:
                        correct +=1
                    total += 1

        return round(correct/total, 4)

def train_network(data):
    model =  FeedForward() # Change during development
    epochs = 50

    print(f"Training {model.__class__.__name__}...")
    m = NNModel(model, 0.0009, data)

    for e in range(epochs):
        print(f"Epoch: {e}/{epochs}")
        m.train_epoch()
        accuracy = m.eval()
        print(f"Test accuracy: {accuracy}")


def output_to_mat(filename, data):
    data_mat = {'M': data}
    sio.savemat(filename, data_mat)

def export_silenced():
    # Export sushant's data.
    for i in range(1,6):
        Silence_combined.load_mat_file('./NEW/sushant/converted/log_'+ str(i) + '.mat')
        silenced_data = Silence_combined.silence_removal(Silence_combined.butterworth())
        output_to_mat('./NEW/sushant/silenced/silenced_'+ str(i) +'.mat', silenced_data)

    # Export soham's data.
    for i in range(1,6):
        Silence_combined.load_mat_file('./NEW/soham/converted/log_'+ str(i) + '.mat')
        silenced_data = Silence_combined.silence_removal(Silence_combined.butterworth())
        output_to_mat('./NEW/soham/silenced/silenced_'+ str(i) +'.mat', silenced_data)

    # Export vintony's data.
    for i in range(1,6):
        Silence_combined.load_mat_file('./NEW/vin/converted/log_'+ str(i) + '.mat')
        silenced_data = Silence_combined.silence_removal(Silence_combined.butterworth())
        output_to_mat('./NEW/vin/silenced/silenced_'+ str(i) +'.mat', silenced_data)

def get_data():
    #matlab_files = ['./NEW/sushant/separated/separated_1.mat', './NEW/soham/separated/separated_1.mat', './NEW/vin/separated/separated_1.mat']

     # Export sushant's data.

    sushant_features = []
    for i in range(1,2):
        Silence_combined.load_mat_file('./NEW/sushant/separated/separated_'+ str(i) + '.mat')
        features = Silence_combined.extract_features()
        sushant_features.extend(features)
    Silence_combined.export_sushant_data(sushant_features)


    soham_features = []
    for i in range(1,2):
        Silence_combined.load_mat_file('./NEW/soham/separated/separated_'+ str(i) + '.mat')
        features = Silence_combined.extract_features()
        soham_features.extend(features)
    Silence_combined.export_soham_data(soham_features)


    # Export vintony's data.
    vintony_features = []
    for i in range(1,2):
        Silence_combined.load_mat_file('./NEW/vin/separated/separated_'+ str(i) + '.mat')
        features = Silence_combined.extract_features()
        vintony_features.extend(features)
    Silence_combined.export_vintony_data(vintony_features)

    return Silence_combined.get_data()

# def get_test_data():
#     # matlab_files = ['./NEW/sushant/converted/log_1.mat', './NEW/soham/converted/log_1.mat', './NEW/vin/converted/log_1.mat']

#     # Export sushant's data.
#     Silence_combined.load_mat_file('./Sushant_CSI/1.3m/forward_converted/log_1.mat')
#     features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
#     Silence_combined.export_sushant_data(features)

#     # Export soham's data.
#     Silence_combined.load_mat_file('./Soham_CSI/converted/log_1.mat')
#     features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
#     Silence_combined.export_soham_data(features)

#     # Export vintony's data.
#     Silence_combined.load_mat_file('./Vintony_CSI/converted/log_1.mat')
#     features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
#     Silence_combined.export_vintony_data(features)

#     return Silence_combined.get_data()


if __name__ == "__main__":

    print("Conducting CSI computation and loading variables..")
    x_train, x_test, y_train, y_test = get_data()
    print("CSI data loaded batching variables...")

    x_train = torch.tensor(x_train.values)
    y_train = torch.tensor(y_train.values)

    num_2 = 0
    num_1 = 0
    num_0 = 0
    for each_val in y_train:
        if each_val == 0:
            num_0 += 1
        elif each_val == 1:
            num_1 += 1
        elif each_val == 2:
            num_2 += 1
        else:
            print("label not heree")

    print("Number of 2s: ", num_2)
    print("Number of 1s: ", num_1)
    print("Number of 0s: ", num_0)

    x_test = torch.tensor(x_test.values)
    y_test = torch.tensor(y_test.values)

    # batching the input to 10 TRAIN DATA
    last_num = math.floor(x_train.shape[0]/10)*10
    x_train = x_train[:last_num]
    x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1])
    x_train = x_train.view(math.floor(x_train.shape[0]/10), 10, 6)

    y_train = y_train[:last_num]
    y_train = y_train.view(math.floor(y_train.shape[0]/10), 10)

    # batching TEST DATA
    last_num = math.floor(x_test.shape[0]/10)*10
    x_test = x_test[:last_num]
    x_test = x_test.view(x_test.shape[0], 1, x_test.shape[1])
    x_test = x_test.view(math.floor(x_test.shape[0]/10), 10, 6)

    y_test = y_test[:last_num]
    y_test = y_test.view(math.floor(y_test.shape[0]/10), 10)

    training = []
    testing = []

    for index, each_train_val in enumerate(x_train):
        training.append((x_train[index], y_train[index]))

    for index, each_test_val in enumerate(x_test):
        testing.append((x_test[index], y_test[index]))

    print("Data batched, training network..")

    train_network([training, testing])

    print("Network training complete.")