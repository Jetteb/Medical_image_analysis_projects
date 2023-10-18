"""
Test code for CAD.
"""

from builtins import print
import numpy as np

import scipy
import scipy.io
from IPython.display import clear_output
    
import cad
import cad_util as util
import registration as reg

import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML


# SECTION 1: Linear regression

def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations
    # E - squared error for the optimal solution

    #------------------------------------------------------------------#
    # TODO: Implement the least-squares solution for w.
    #------------------------------------------------------------------#

    w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # compute the error
    E = np.transpose(A.dot(w) - b).dot(A.dot(w) - b)

    return w, E

def linear_regression():
    
    # load the training, validation and testing datasets
    fn1 = '../data/linreg_ex_test.txt'
    fn2 = '../data/linreg_ex_train.txt'
    fn3 = '../data/linreg_ex_validation.txt'
    # shape (30,2) numpy array; x = column 0, y = column 1
    test_data = np.loadtxt(fn1)
    # shape (20,2) numpy array; x = column 0, y = column 1
    train_data = np.loadtxt(fn2)
    # shape (10,2) numpy array; x = column 0, y = column 1
    validation_data = np.loadtxt(fn3)

    # plot the training dataset
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # ax.plot(train_data[:,0], train_data[:,1], '*')
    # ax.grid()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Training data')

    #---------------------------------------------------------------------#
    # TODO: Implement training of a linear regression model.
    # Here you should reuse ls_solve() from the registration mini-project.
    # The provided addones() function adds a column of all ones to a data
    # matrix X in a similar way to the c2h() function used in registration.

    trainX = train_data[:,0].reshape(-1,1)
    trainY = train_data[:,1].reshape(-1,1)
    
    trainX_1 = util.addones(trainX)
    
    theta, E = ls_solve(trainX_1, trainY)
    
    #---------------------------------------------------------------------#

    fig1 = plt.figure(figsize=(5,5))
    ax1 = fig1.add_subplot(111)
    util.plot_regression(trainX, trainY, theta, ax1)
    ax1.grid()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax1.set_title('Training set')

    testX = test_data[:,0].reshape(-1,1)
    testY = test_data[:,1].reshape(-1,1)
    
    fig2 = plt.figure(figsize=(5,5))
    ax2 = fig2.add_subplot(111)
    util.plot_regression(testX, testY, theta, ax2)
    ax2.grid()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax2.set_title('Test set')

    #---------------------------------------------------------------------#
    # TODO: Compute the error for the trained model.
    #---------------------------------------------------------------------#
    validationX = validation_data[:,0].reshape(-1,1)
    validationY = validation_data[:,1].reshape(-1,1)
    
    validationX = util.addones(validationX)
    
    E_validation = np.sum((validationX.dot(theta) - validationY) ** 2) / (2 * len(validationY))

    testX = util.addones(testX)

    E_test = np.sum((testX.dot(theta) - testY) ** 2) / (2 * len(testY))

    return E_validation, E_test


def quadratic_regression():
    # load the training, validation and testing datasets
    fn1 = '../data/linreg_ex_test.txt'
    fn2 = '../data/linreg_ex_train.txt'
    fn3 = '../data/linreg_ex_validation.txt'
    # shape (30,2) numpy array; x = column 0, y = column 1
    test_data = np.loadtxt(fn1)
    # shape (20,2) numpy array; x = column 0, y = column 1
    train_data = np.loadtxt(fn2)
    # shape (10,2) numpy array; x = column 0, y = column 1
    validation_data = np.loadtxt(fn3)

    # plot the training dataset
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # ax.plot(train_data[:,0], train_data[:,1], '*')
    # ax.grid()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Training data')

    #---------------------------------------------------------------------#
    # TODO: Implement training of a linear regression model.
    # Here you should reuse ls_solve() from the registration mini-project.
    # The provided addones() function adds a column of all ones to a data
    # matrix X in a similar way to the c2h() function used in registration.

    trainX = train_data[:,0].reshape(-1,1)
    trainY = train_data[:,1].reshape(-1,1)
    
    x_quad = np.hstack((trainX, trainX**2))
    x_quad_1 = util.addones(x_quad)
    
    theta, E = ls_solve(x_quad_1, trainY)
    
    #---------------------------------------------------------------------#

    fig1 = plt.figure(figsize=(5,5))
    ax1 = fig1.add_subplot(111)
    util.plot_regression(x_quad, trainY, theta, ax1)
    ax1.grid()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax1.set_title('Training set')

    testX = test_data[:,0].reshape(-1,1)
    testY = test_data[:,1].reshape(-1,1)
    
    x_quad_test = np.hstack((testX, testX**2))
    x_quad_1_test = util.addones(x_quad_test)

    fig2 = plt.figure(figsize=(5,5))
    ax2 = fig2.add_subplot(111)
    util.plot_regression(x_quad_test, testY, theta, ax2)
    ax2.grid()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax2.set_title('Test set')

    #---------------------------------------------------------------------#
    # TODO: Compute the error for the trained model.
    #---------------------------------------------------------------------#
    

    validationX = validation_data[:,0].reshape(-1,1)
    validationY = validation_data[:,1].reshape(-1,1)
    
    x_quad_validation = np.hstack((validationX, validationX**2))
    x_quad_1_validation = util.addones(x_quad_validation)

    
    E_validation = np.sum((x_quad_1_validation.dot(theta) - validationY) ** 2) / (2 * len(validationY))

    E_test = np.sum((x_quad_1_test.dot(theta) - testY) ** 2) / (2 * len(testY))

    return E_validation, E_test


# SECTION 2: Logistic regression
 

def logistic_regression(mu, batch_size, num_iterations):
    
    # dataset preparation
    num_training_samples = 300
    num_validation_samples = 100
    
    # here we reuse the function from the segmentation practicals
    m1=[2,3]
    m2=[-0,-4]
    s1=[[8,7],[7,8]]
    s2=[[8,6],[6,8]]

    [trainingX, trainingY] = util.generate_gaussian_data(num_training_samples, m1, m2, s1, s2)
    r,c = trainingX.shape
    print('Training sample shape: {}'.format(trainingX.shape))

    # we need a validation set to monitor for overfitting
    [validationX, validationY] = util.generate_gaussian_data(num_validation_samples, m1, m2, s1, s2)
    r_val, c_val = validationX.shape
    print('Validation sample shape: {}'.format(validationX.shape))
    
    validationXones = util.addones(validationX)

    # train a logistic regression model:
    # the learning rate for the gradient descent method
    # (the same as in intensity-based registration)
    mu = 0.001

    # we are actually using stochastic gradient descent
    batch_size = 100

    # initialize the parameters of the model with small random values,
    # we need one parameter for each feature and a bias
    Theta = 0.02*np.random.rand(c+1, 1)

    # number of gradient descent iterations
    num_iterations = 300

    # variables to keep the loss and gradient at every iteration
    # (needed for visualization)
    iters = np.arange(num_iterations)
    loss = np.full(iters.shape, np.nan)
    validation_loss = np.full(iters.shape, np.nan)

    # Create base figure
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(121)
    im1, Xh_ones, num_range_points = util.plot_lr(trainingX, trainingY, Theta, ax1)
    util.scatter_data(trainingX, trainingY, ax=ax1)
    ax1.grid()
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.legend()
    ax1.set_title('Training set')
    text_str1 = '{:.4f};  {:.4f};  {:.4f}'.format(0, 0, 0)
    txt1 = ax1.text(0.3, 0.95, text_str1, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax1.transAxes)

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    ax2.set_title('mu = '+str(mu))
    h1, = ax2.plot(iters, loss, linewidth=2, label='Training loss')
    h2, = ax2.plot(iters, validation_loss, linewidth=2, label='Validation loss')
    ax2.set_ylim(0, 0.7)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()
    ax1.legend()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    # iterate
    for k in np.arange(num_iterations):
        
        # pick a batch at random
        idx = np.random.randint(r, size=batch_size)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(util.addones(trainingX[idx,:]), trainingY[idx], Theta)

        # gradient descent:
        # here we reuse the code for numerical computation of the gradient
        # of a function
        Theta = Theta - mu*reg.ngradient(loss_fun, Theta)

        # compute the loss for the current model parameters for the
        # training and validation sets
        # note that the loss is divided with the number of samples so
        # it is comparable for different number of samples
        loss[k] = loss_fun(Theta)/batch_size
        validation_loss[k] = cad.lr_nll(validationXones, validationY, Theta)/r_val

        # upldate the visualization
        ph = cad.sigmoid(Xh_ones.dot(Theta)) > 0.5
        decision_map = ph.reshape(num_range_points, num_range_points)
        decision_map_trns = np.flipud(decision_map)
        im1.set_data(decision_map_trns)
        text_str1 = '{:.4f};  {:.4f};  {:.4f}'.format(Theta[0,0], Theta[1,0], Theta[2,0])
        txt1.set_text(text_str1)
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.={}, loss={:.3f}, val. loss={:.3f} '.format(k, loss[k], validation_loss[k])
        txt2.set_text(text_str2)

        plt.pause(0.001)
        display(fig)
        clear_output(wait = True)
        
    
    return

        
# SECTION 3: Building blocks of neural networks
 
def model_training():

    # Define inputs
    x = 1.5   # input
    y = 0.5   # desired output
    w = 0.8   # initial weight
    r = 0.1   # learning rate

    # Create list with loss values, start with initial loss
    L = [(x*w - y)**2]

    # Print the values that need to be filled in in the table
    print('Epoch\t\tWeight\t\tPredicted')
    print(f'0\t\t{w:.5f}\t\t{x*w:.5f}')

    # Train model for 20 epochs
    for epoch in range(1,21):
        # Calculate gradient
        #---------------------------------------------------------------------#
        # TODO: Define the derivative of L with respect to w (as a function of
        # w, x, and y). Implement it as follows:
        # dL_dw = ...
        #---------------------------------------------------------------------#
        y_hat = x*w
        dL_dw = 2*x*(y_hat - y)
        
        # Take a step and update the weight
        w = w - r*dL_dw

        # Calculate new loss
        loss = (x*w - y)**2
        L.append(loss)

        # Print the values of the weight and predicted output   
        print(f'{epoch}\t\t{w:.5f}\t\t{x*w:.5f}')

    # Plot Loss curve of training data    
    plt.figure()
    plt.plot(range(len(L)), L)
    plt.title('Loss value over time')
    plt.xticks(range(0,len(L),2))
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.show()
    
class n_hidden_neural_network:
    
    def data_preprocessing(self):

        ## load dataset (images and labels y)
        fn = '../data/nuclei_data_classification.mat'
        mat = scipy.io.loadmat(fn)

        training_images = mat["training_images"]     # (24, 24, 3, 14607)
        self.training_y = mat["training_y"]          # (14607, 1)

        validation_images = mat["validation_images"] # (24, 24, 3, 7303)
        self.validation_y = mat["validation_y"]      # (7303, 1)

        test_images = mat["test_images"]             # (24, 24, 3, 20730)
        self.test_y = mat["test_y"]                  # (20730, 1)

        ## dataset preparation
        # Reshape matrices and normalize pixel values
        self.training_x, self.validation_x, self.test_x = util.reshape_and_normalize(training_images, 
                                                                                     validation_images, 
                                                                                     test_images)      

        # Visualize several training images classified as large or small
        #util.visualize_big_small_images(self.training_x, self.training_y, training_images.shape)
        #util.visualize_big_small_images(training_images, self.training_y)

    # Initialize the weights of the model according to the shapes that are given for the network.
    # We initialize all weights with random numbers that are then multiplied by 2 and subtracted by one.
    # This method was also used in the given network. The weights are then returned
    def init_model(self, w_shapes):

        weights = {}
        for i, shape in enumerate(w_shapes):
             weights[f'w{i+1}']  = np.random.rand(shape[0], shape[1])*2-1

        return weights 
            
    # In this function the shapes of the network will be defined. They will have X amount
    # of hidden features given by the variable hidden_features. This would make a layer have shape (100,100)
    #  in the case of 100 hidden features. The amount of layers is defined by num_hidden_layers.
    # The first layer of the network takes on the shape of the training data and the hidden_features, making the first layer (shape(training_data), hidden_features)
    # Then, the amount of hidden layers-1 is added. This is because in order to make this a classification we need to scale down to a matrix (1,1) and therefore we need
    # 2 layers before this of shape (hidden_features, 1) to create this. Then, these shapes are converted to weight matrices in the function init_model. the weights are returned
    def define_shapes(self,  hidden_features, num_hidden_layers):
        in_features = self.training_x.shape[1]
        out_features = 1  
        
        w_shapes = []
        w_shapes.append((in_features, hidden_features))
        for x in range(num_hidden_layers-1):
            w_shapes.append((hidden_features, hidden_features))
        w_shapes.append((hidden_features, 1))
        w_shapes.append((hidden_features, 1))

        
        weights =self.init_model(w_shapes)
        
        return weights
        
    # This function is the same as the original provided function, so it trains the network. The changes are parameter changes for define_shapes, forward and backward.
    # Furthermore, the parameters of the network (learing_rate, batchsize, n_hidden_features, num_hidden_layers) can now be changed by the user.
    
    def launch_training(self, learning_rate, batchsize, hidden_features, num_hidden_layers, epochs = 100):
        
        
        self.learning_rate = learning_rate #0.001
        self.batchsize = batchsize #128
        n_hidden_features = hidden_features #1000
        self.num_hidden_layers = num_hidden_layers
        
        # Define empty lists for saving training progress variables
        training_loss = []
        validation_loss = []
        Acc = []
        steps = []

        # randomly initialize model weights
        self.weights = self.define_shapes(hidden_features, self.num_hidden_layers)

        print('> Start training ...')
        # Train for n_epochs epochs
        n_epochs = epochs
        for epoch in range(n_epochs): 

            # Shuffle training images every epoch
            training_x, training_y = util.shuffle_training_x(self.training_x, self.training_y)

            for batch_i in range(self.training_x.shape[0]//self.batchsize):

                ## sample images from this batch
                batch_x = training_x[self.batchsize*batch_i : self.batchsize*(batch_i+1)]
                batch_y = training_y[self.batchsize*batch_i : self.batchsize*(batch_i+1)]

                ## train on one batch
                # Forward pass
                hidden, output = self.forward(batch_x, self.weights, self.num_hidden_layers)
                # Backward pass    
                self.weights = self.backward(batch_x, batch_y, output, hidden, self.weights, learning_rate, num_hidden_layers)

                ## Save values of loss function for plot
                training_loss.append(util.loss(output, batch_y))
                steps.append(epoch + batch_i/(self.training_x.shape[0]//self.batchsize))

            ## Validation images trhough network
            # Forward pass only (no backward pass in inference phase!)
            _, val_output = self.forward(self.validation_x, self.weights, self.num_hidden_layers)
            # Save validation loss
            val_loss = util.loss(val_output, self.validation_y)
            validation_loss.append(val_loss)
            accuracy = (self.validation_y == np.round(val_output)).sum()/(self.validation_y.shape[0])
            Acc.append(accuracy*100)

            # Plot loss function and accuracy of validation set
            clear_output(wait=True)
            fig, ax = plt.subplots(1,2, figsize=(15,10))
            ax[0].plot(steps,training_loss)
            ax[0].plot(range(1, len(validation_loss)+1), validation_loss, '.')
            ax[0].legend(['Training loss', 'Validation loss'])
            ax[0].set_title(f'Loss curves after {epoch+1}/{n_epochs} epochs')
            ax[0].set_ylabel('Loss'); ax[0].set_xlabel('epochs')
            ax[0].set_xlim([0, epochs]); ax[0].set_ylim([0, max(training_loss)])
            ax[1].plot(Acc)
            ax[1].set_title(f'Validation accuracy after {epoch+1}/{n_epochs} epochs')
            ax[1].set_ylabel('Accuracy'); ax[1].set_xlabel('epochs')
            ax[1].set_xlim([0, epochs]); ax[1].set_ylim([0, 100])
            plt.show()
        print('> Training finished')
            
    # Same as in the provided function. Computes the test accuracy of the model.
    def pass_on_test_set(self):
        
        # Forward pass on test set
        _, test_output = self.forward(self.test_x, self.weights,self.num_hidden_layers)
        test_accuracy = (self.test_y == np.round(test_output)).sum()/(self.test_y.shape[0])
        print('Test accuracy: {:.2f}'.format(test_accuracy))

        # Plot final test predictions
        large_list = test_output[self.test_y==1]
        small_list = test_output[self.test_y==0]
        plt.figure()
        plt.hist(small_list, 50, alpha = 0.5)
        plt.hist(large_list, 50, alpha = 0.5)
        plt.legend(['Small (label = 0)','Large (label = 1)'], loc = 'upper center')
        plt.xlabel('Prediction')
        plt.title('Final test set predictions')
        plt.show()
        
        return test_accuracy
        
   # the forward function is modified such that it can handle an adaptive number of hidden layers. First we say that the hidden_layers parameter containing the forwarding
   # of the network to contain x, the training data which is essentially where the network starts. Then, this is passed through the number of hidden layers where of the weights and the previous hidden layer
   # the dot product is taken and put into the sigmoid function, much like in the original network. Then, the output is created by taking the last sigmoid of the weights with the last hidden_layer.
   # These are finally returned.
    def forward(self, x, weights, num_hidden_layers):
        hidden_layers = [x]
        
        for i in range(num_hidden_layers):
            w = weights[f'w{i+1}']
            hidden = util.sigmoid(np.dot(hidden_layers[-1], w))
            hidden_layers.append(hidden)
        
        output = util.sigmoid(np.dot(hidden_layers[-1], weights[f'w{num_hidden_layers+1}']))
        return hidden_layers, output
    
    # The backward function now takes the derivative of the differnt formulas of the layers and by doing this loops backwards over the layers.
    # We first define the gradient list which is the amount of layers (hidden+output+input). Then, the derivative is taken from the output and saved in the gradient array needed for the updating.
    # Then, we loop over the amount of hidden layers again and compute all the derivatives of the layers and save the gradients. Finally, the last gradient is taken of the input layer and all the layers
    # are then updated by multiplying the gradients with the learning rate and adding this change to the weight. These weights are then returned.
    def backward(self, x, y, output, hidden_layers, weights, learning_rate, num_hidden_layers):
        
        num_layers = num_hidden_layers + 1
        gradients = [None] * (num_layers+1)

        delta_output = 2 * (output - y) * util.sigmoid_derivative(output)
        gradients[num_layers] = np.dot(hidden_layers[-1].T, delta_output)

        delta_hidden = delta_output
        for layer in range(num_hidden_layers, 0, -1):
            w_next = weights[f'w{layer+1}']
            delta_hidden = np.dot(delta_hidden, w_next.T) * util.sigmoid_derivative(hidden_layers[layer])
            gradients[layer] = np.dot(hidden_layers[layer-1].T, delta_hidden)

        delta_hidden = np.dot(delta_hidden, weights['w2'].T) * util.sigmoid_derivative(hidden_layers[-1])
        gradients[0] = np.dot(x.T, delta_hidden)

        for layer in range(1, num_layers + 1):
            weights[f'w{layer}'] -= learning_rate * gradients[layer]

        return weights


    
class Training:
    
    def data_preprocessing(self):

        ## load dataset (images and labels y)
        fn = '../data/nuclei_data_classification.mat'
        mat = scipy.io.loadmat(fn)

        training_images = mat["training_images"]     # (24, 24, 3, 14607)
        self.training_y = mat["training_y"]          # (14607, 1)

        validation_images = mat["validation_images"] # (24, 24, 3, 7303)
        self.validation_y = mat["validation_y"]      # (7303, 1)

        test_images = mat["test_images"]             # (24, 24, 3, 20730)
        self.test_y = mat["test_y"]                  # (20730, 1)

        ## dataset preparation
        # Reshape matrices and normalize pixel values
        self.training_x, self.validation_x, self.test_x = util.reshape_and_normalize(training_images, 
                                                                                     validation_images, 
                                                                                     test_images)      

        # Visualize several training images classified as large or small
        #util.visualize_big_small_images(self.training_x, self.training_y, training_images.shape)
        #util.visualize_big_small_images(training_images, self.training_y)

    def define_shapes(self, learning_rate, batchsize, hidden_features):

        self.learning_rate = learning_rate #0.001
        self.batchsize = batchsize #128
        n_hidden_features = hidden_features #1000

        in_features = self.training_x.shape[1]
        out_features = 1                  # Classification problem, so you want to obtain 1 value (a probability) per image

        # Define shapes of the weight matrices
        #---------------------------------------------------------------------#
        # TODO: Create two variables: w1_shape and w2_shape, and define them as
        # follows (as a function of variables defined above)
        # self.w1_shape = (.. , ..)
        # self.w2_shape = (.. , ..)
        #---------------------------------------------------------------------#
        
        self.w1_shape = (in_features, n_hidden_features)
        self.w2_shape = (n_hidden_features, out_features)
        
        return {'w1_shape': self.w1_shape,
                'w2_shape': self.w2_shape}

    def launch_training(self, epochs = 100):
        
        # Define empty lists for saving training progress variables
        training_loss = []
        validation_loss = []
        Acc = []
        steps = []

        # randomly initialize model weights
        self.weights = util.init_model(self.w1_shape, self.w2_shape)

        print('> Start training ...')
        # Train for n_epochs epochs
        n_epochs = epochs
        for epoch in range(n_epochs): 

            # Shuffle training images every epoch
            training_x, training_y = util.shuffle_training_x(self.training_x, self.training_y)

            for batch_i in range(self.training_x.shape[0]//self.batchsize):

                ## sample images from this batch
                batch_x = training_x[self.batchsize*batch_i : self.batchsize*(batch_i+1)]
                batch_y = training_y[self.batchsize*batch_i : self.batchsize*(batch_i+1)]

                ## train on one batch
                # Forward pass
                hidden, output = self.forward(batch_x, self.weights)
                # Backward pass    
                self.weights = self.backward(batch_x, batch_y, output, hidden, self.weights)

                ## Save values of loss function for plot
                training_loss.append(util.loss(output, batch_y))
                steps.append(epoch + batch_i/(self.training_x.shape[0]//self.batchsize))

            ## Validation images trhough network
            # Forward pass only (no backward pass in inference phase!)
            _, val_output = self.forward(self.validation_x, self.weights)
            # Save validation loss
            val_loss = util.loss(val_output, self.validation_y)
            validation_loss.append(val_loss)
            accuracy = (self.validation_y == np.round(val_output)).sum()/(self.validation_y.shape[0])
            Acc.append(accuracy)

            # Plot loss function and accuracy of validation set
        # clear_output(wait=True)
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        ax[0].plot(steps,training_loss)
        ax[0].plot(range(1, len(validation_loss)+1), validation_loss, '.')
        ax[0].legend(['Training loss', 'Validation loss'])
        ax[0].set_title(f'Loss curves after {epoch+1}/{n_epochs} epochs')
        ax[0].set_ylabel('Loss'); ax[0].set_xlabel('epochs')
        ax[0].set_xlim([0, epochs]); ax[0].set_ylim([0, max(training_loss)])
        ax[1].plot(Acc)
        ax[1].set_title(f'Validation accuracy after {epoch+1}/{n_epochs} epochs')
        ax[1].set_ylabel('Accuracy'); ax[1].set_xlabel('epochs')
        ax[1].set_xlim([0, epochs]); ax[1].set_ylim([min(Acc),0.8])
        plt.show()
        print('> Training finished')
        
            
    def pass_on_test_set(self):
        
        # Forward pass on test set
        _, test_output = self.forward(self.test_x, self.weights)
        test_accuracy = (self.test_y == np.round(test_output)).sum()/(self.test_y.shape[0])
        print('Test accuracy: {:.2f}'.format(test_accuracy))

        # Plot final test predictions
        large_list = test_output[self.test_y==1]
        small_list = test_output[self.test_y==0]
        plt.figure()
        plt.hist(small_list, 50, alpha = 0.5)
        plt.hist(large_list, 50, alpha = 0.5)
        plt.legend(['Small (label = 0)','Large (label = 1)'], loc = 'upper center')
        plt.xlabel('Prediction')
        plt.title('Final test set predictions')
        plt.show()
            
        return test_accuracy
        
    def forward(self, x, weights):
        w1 = weights['w1']
        w2 = weights['w2']

        hidden = util.sigmoid(np.dot(x, w1))
        output = util.sigmoid(np.dot(hidden, w2))

        return hidden, output

    def backward(self, x, y, output, hidden, weights):
        w1 = weights['w1']
        w2 = weights['w2']
        

        dL_dw2 = np.dot(hidden.T, (2*(output - y) * util.sigmoid_derivative(output))) 
        dL_dw1 = np.dot(x.T,  (np.dot(2*(output - y) * util.sigmoid_derivative(output), w2.T) * util.sigmoid_derivative(hidden)))

        #---------------------------------------------------------------------#
        # TODO: Update the variables: w1 and w2, and define them as
        # follows (as a function of learning_rate, dL_dw1, and dL_dw2)
        # w1 = w1 - ...
        # w2 = w2 - ...
        #---------------------------------------------------------------------#
        
        w1 = w1 - (self.learning_rate*dL_dw1)
        w2 = w2 - (self.learning_rate*dL_dw2)
        
        
        return {'w1': w1,
                'w2': w2}
    

# Section 4: Unsupervised learning, PCA

def covariance_matrix_test():
    N=100
    mu1=[0,0]
    mu2=[0,0]
    sigma1=[[3,1],[1,1]]
    sigma2=[[3,1],[1,1]]
    X, Y = util.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)
    
    data_mean = np.mean(X, axis=0)
    
    data_covariance = np.cov(X, rowvar=False)
    
   # Print the results
    print("Input Mean for Class 1:", mu1)
    print("Input Mean for Class 2:", mu2)
    print("Input Covariance Matrix for Class 1:")
    print(sigma1)
    print("Input Covariance Matrix for Class 2:")
    print(sigma2)
    print("\nCalculated Mean of the Data:")
    print(data_mean)
    print("\nCalculated Covariance Matrix of the Data:")
    print(data_covariance)
    #------------------------------------------------------------------#
    # TODO: Calculate the mean and covariance matrix of the data,
    #  and compare them to the parameters you used as input.
    #------------------------------------------------------------------#
    return X, Y, [sigma1, sigma2]

def eigen_vecval_test(sigma):
    #------------------------------------------------------------------#
    # TODO: Compute the eigenvectors and eigenvalues of the covariance matrix,
    #  what two properties can you name about the eigenvectors? How can you verify these properties?
    #  which eigenvalue is the largest and which is the smallest?
    #------------------------------------------------------------------#
    
    w, v = np.linalg.eig(sigma)
    ix = np.argsort(w)[::-1] #Find ordering of eigenvalues
    w = w[ix] #Reorder eigenvalues
    v = v[:, ix]
    
   # Check the first property (unit vectors)
    eigenvector_magnitudes = np.linalg.norm(v, axis=0)
    print("Eigenvector Magnitudes:")
    print(eigenvector_magnitudes)

    # Check the second property (orthogonality)
    dot_product_matrix = np.dot(v.T, v)
    print("Dot Product Matrix (should be close to identity matrix):")
    print(dot_product_matrix)

    # Print eigenvalues
    print("\nEigenvalues (largest to smallest):")
    print(w)
    
    return v, w

def rotate_using_eigenvectors_test(X, Y, v):
    #------------------------------------------------------------------#
    # TODO: Rotate X using the eigenvectors
    #------------------------------------------------------------------#
    # Convert X and Y to NumPy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Convert v to a NumPy array
    v = np.array(v)

    # Create a rotation matrix using the eigenvectors
    rotation_matrix = np.column_stack((v, v[::-1]))  # Assuming v is a 2D vector


    # Rotate the data using the rotation matrix
    rotated_data_X = np.dot(X, rotation_matrix)
    # rotated_data_Y = np.dot(Y, rotation_matrix)

    return rotated_data_X #, rotated_data_Y

def test_mypca():
    #Generates some toy data in 2D, computes PCA, and plots both datasets
    N=100
    mu1=[0,0]
    mu2=[2,0]
    sigma1=[[2,1],[1,1]]
    sigma2=[[2,1],[1,1]]

    XG, YG = util.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    fig = plt.figure(figsize=(15,6))

    ax1 = fig.add_subplot(121)
    util.scatter_data(XG,YG,ax=ax1)
    sigma = np.cov(XG, rowvar=False)
    w, v = np.linalg.eig(sigma)
    ax1.plot([0, v[0,0]], [0, v[1,0]], c='g', linewidth=3, label='Eigenvector1')
    ax1.plot([0, v[0,1]], [0, v[1,1]], c='k', linewidth=3, label='Eigenvector2')
    ax1.set_title('Original data')
    ax_settings(ax1)

    ax2 = fig.add_subplot(122)
    X_pca, v, w, fraction_variance = cad.mypca(XG)
    util.scatter_data(X_pca,YG,ax=ax2)
    sigma2 = np.cov(X_pca, rowvar=False)
    w2, v2 = np.linalg.eig(sigma2)
    ax2.plot([0, v2[0,0]], [0, v2[1,0]], c='g', linewidth=3, label='Eigenvector1')
    ax2.plot([0, v2[0,1]], [0, v2[1,1]], c='k', linewidth=3, label='Eigenvector2')
    ax2.set_title('My PCA')
    ax_settings(ax2)

    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), bbox_transform=plt.gcf().transFigure, ncol = 4)

    print(fraction_variance)

def ax_settings(ax):
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()


if __name__ == "__main__":

    logistic_regression()