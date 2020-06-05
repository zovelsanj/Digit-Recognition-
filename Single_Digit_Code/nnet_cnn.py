import _pickle as c_pickle, gzip
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model, Flatten

def main():
    # ======= Load the dataset =========== 
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # We need to rehape the data back into a 1x28x28 image to make it a 4D tensor
    # as Conv2d() takes input parameters from 4D tensor
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))
    # print(X_train.shape)
    # print(X_test.shape)

    # =========== Split into train(90%) and dev(10%) for validation set =========
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # ======== Split dataset into batches ==========
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #### ============ MODEL SPECIFICATION ==================
    model = nn.Sequential(
              nn.Conv2d(1, 32, (3, 3)),         #A convolutional layer with 32 filters of size  3×3, in_channel=1
              nn.ReLU(),                        #A ReLU nonlinearity
              nn.MaxPool2d((2, 2)),             #A max pooling layer with size  2×2

              nn.Conv2d(32,64,(3,3)),           #A convolutional layer with 64 filters of size  3×3
              nn.ReLU(),
              nn.MaxPool2d((2,2)),

              Flatten(),                        #A flatten layer
              nn.Linear(1600,128),              #A fully connected layer with 128 neurons
              nn.Dropout(0.5),                  #A dropout layer with drop probability 0.5
              nn.Linear(128,10),                #A fully-connected layer with 10 neurons
            )
    
    # Use the nn package to define our model as a sequence of layers. 
    # nn.Sequential is a Module which contains other Modules, and applies them in sequence to produce its output.

    # nn.Conv2d(input image channel, output channel, (square convolution or Kernel size))
    # nn.Maxpool2d((Kernel size),(stride)) 
    # Kernel size =>  the size of the window to take a max over
    # stride – the stride of the window. Default value is Kernel size. stride = (2,2) halves the image dimension output from Conv2d
    # nn.Linear (in features, out features, bias=True)
    # in features => size of each input sample
    # out features => size of each output sample
    # nn.Dropout(Drop out probability)
    # Dropout is a regularization technique that “drops out” or “deactivates” few neurons in the neural network randomly in order to avoid the problem of overfitting.  
    ##################################

    val_acc, train_acc, train_loss, v_loss, v_acc=train_model(train_batches, dev_batches, model, nesterov=True)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))
    
    # epoch = np.arange(1,11)
    # fig, axs = plt.subplots(2)
    # axs[0].plot(epoch,train_acc)
    # axs[0].plot(epoch,v_acc)
    # axs[0].set_title('Train vs. Validation Accuracy')
    # axs[0].set(xlabel='Number of Epoches', ylabel='Accuracy')
    
    # axs[1].plot(epoch,train_loss)
    # axs[1].plot(epoch,v_loss)
    # axs[1].set_title('Train vs. Validation Loss')
    # axs[1].set(xlabel='Number of Epoches', ylabel='Loss')

    # fig.tight_layout(pad=0.5)
    # plt.legend([axs[0], axs[1]], ["Train_Accuracy Validation_Accuracy", "Train_Loss Validation_Loss"])
    # plt.show()
    

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. 
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)
    main()
    
