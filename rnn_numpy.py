import numpy as np
import operator
import datetime
import sys
import prepare_data



class RNNNumpy:
    def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):
        self.word_dim=word_dim
        self.hidden_dim=hidden_dim
        self.bptt_truncate=bptt_truncate

        #Randomly initialize the network parameters
        #Acutally there are 2*word_dim*hidden_dim+ hidden_dim*hidden_dim parameters
        self.U= np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        self.V= np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def softmax(x):
        """Compute the softmax in a numerically stable way."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def forward_propagation(self,x):
        """

        :param x: array of T word index which is from 0 to self.hidden_dim(not one-hot vector)
        :return: [o,s]  o contains output, s contains the temporary states
        """

        #The total number of time steps
        T= len(x)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which is set to 0

        s=np.zeros((T+1,self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        # The outputs at each time step. Again, we save them for later
        o=np.zeros((T,self.word_dim))

        for t in range(T):
            # Note that we are indexing U by x[t]. This is the same as multiplying U with a one-hot vector
            s[t]=np.tanh(self.U[:,x[t]]+self.W.dot(s[t-1]))
            o[t]=RNNNumpy.softmax(self.V.dot(s[t]))

        return [o,s]

    def predict(self,x):
        # Perform forward propagation and return index of the highest score
        o,s=self.forward_propagation(x)
        return np.argmax(o,axis=1)

    def calculate_total_loss(self,x,y):
        L=0

        #For each sentence
        for i in np.arange(len(y)):
            o,s=self.forward_propagation(x[i])

            # We only care about our prediction of the "correct" words
            correct_word_predictions=o[np.arange(len(y[i])),y[i]]

            # cross entropy
            L += -1 * np.sum(np.log(correct_word_predictions))

        return L

    def calculate_loss(self,x,y):
        N= sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

    def bptt(self,x,y):
        T=len(y)
        # Perform forward propagation
        o,s=self.forward_propagation(x)
        # Accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdV = np.zeros(self.V.shape)
        delta_o = o
        delta_o[np.arange(len(y)),y] -= 1.

        # For each output backwards ...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t],s[t].T)

            # Initial delta calculation
            delta_t =self.V.T.dot(delta_o[t])*(1-(s[t]**2))

            # Back propagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0,t-self.bptt_truncate),t+1)[::-1]:
                dLdW += np.outer(delta_t,s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self,x,y,h=0.001,error_threshold=0.01):
        # Calculate the gradients using BP, we want to check if these are correct
        bptt_gradients=self.bptt(x,y)
        model_params=['U','V','W']
        for pidx,pname in enumerate(model_params):
            parameter=operator.attrgetter(pname)(self)
            print('Performing gradient check for parameter %s with size %d.'%(pname,np.prod(parameter.shape)))

            #Iterate over each element of the parameter matrix, e.g. (0,0),(0,1), ...
            it=np.nditer(parameter,flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix=it.multi_index
                # Save the original value so we can reset it later
                original_value=parameter[ix]

                # Estimate the gradient using (f(x+h)-f(x-h))/(2*h)
                parameter[ix]=original_value+h
                gradplus=self.calculate_total_loss([x],[y])
                parameter[ix]=original_value-h
                gradminus=self.calculate_total_loss([x],[y])
                estimated_gradient=(gradplus-gradminus)/(2*h)

                # Reset parameter to original value
                parameter[ix] = original_value

                # The gradient for this parameter calculated using backpropagation
                backprop_gradient=bptt_gradients[pidx][ix]

                # calculate the relative error(|x-y|/(|x|+|y|))

                denominator=np.abs(backprop_gradient) + np.abs(estimated_gradient)
                relative_error=np.abs(backprop_gradient-estimated_gradient)/denominator if denominator >0 else 0

                if relative_error >error_threshold:
                    print("Gradient check ERROR: parameter=%s ix=%s"%(pname,ix))
                    print("+h Loss:%f"%gradplus)
                    print("-h Loss:%f" % gradminus)
                    print("Estimated_gradient: %f"% estimated_gradient)
                    print("Backpropagation_gradient: %f" % backprop_gradient)
                    print("Relative ERROR:%f"%relative_error)
                    return
                it.iternext()
            print("Gradient check for paramenter %s passed."%(pname))

    # Perform one step of SGD
    def numpy_sdg_step(self,x,y,learning_rate):
        # Calculate the gradients
        dLdU,dLdV,dLdW =self.bptt(x,y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # Outer SGD Loop
    # - model: The Rnn model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - epochs: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self,X_train,y_train,learning_rate=0.005,epochs=100, evaluate_loss_after=5):
        losses=[]
        num_examples_seen=0
        for epoch in range(epochs):
            # Optionally evaluate the loss
            if epoch %evaluate_loss_after==0:
                loss=self.calculate_loss(X_train,y_train)
                losses.append((num_examples_seen,loss))
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('%s: Loss after num_examples_seen=%d epoch=%d: %f'%(time,num_examples_seen,epoch,loss))
                if len(losses)>1 and (losses[-1][1]>losses[-2][1]):
                    learning_rate=0.5*learning_rate
                    print('Setting learning rate to %f'%learning_rate)
                sys.stdout.flush()

            # For each training example ...
            for i in range(len(y_train)):
                self.numpy_sdg_step(X_train[i],y_train[i],learning_rate)
                num_examples_seen+=1

        return losses


if __name__=='__main__':
    src_root = '/Users/jinzixiang/Documents/workspace/python/rnn'
    vocabulary_size = 4000
    X_train,y_train =prepare_data.get_data(src_root,vocabulary_size)
    np.random.seed(10)




    model=RNNNumpy(vocabulary_size)
    predictions=model.predict(X_train[0][0:100])
    print(predictions)

    print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
    print("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))

    grade_check_vocabulary_size=100
    model=RNNNumpy(grade_check_vocabulary_size,hidden_dim=10,bptt_truncate=1000)
    model.gradient_check([0,1,2,3],[1,2,3,4])

    model = RNNNumpy(vocabulary_size)
    losses = model.train_with_sgd( X_train[:100], y_train[:100], epochs=10, evaluate_loss_after=1)
    print(losses)


