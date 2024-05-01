'''
    Problem Statement - 
        Implement ANN tarining process by using forward and backward propagation.
        
    Explanation - 
        The problem we're trying to solve here is training a computer program to recognize the ANDNOT logic. ANDNOT is a binary operation that gives a true output only when the inputs are (one is 1 and the other is 0). For any other combination (both 0s or both 1s), the output is false.
'''



import numpy as np 

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))
    
    
 
inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])
    
target = np.array([
    [0],
    [0],
    [1],
    [0],
])
        
# Define network parameters
input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1
learn_rate = 0.1
epochs = 10000


hidden_weights = np.random.uniform(size=(input_layer_neurons,hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1,hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons,output_layer_neurons))
output_bias = np.random.uniform(size=(1,output_layer_neurons))
        

for epoch in range(epochs):
    
    hidden_layer_sum = np.dot(inputs,hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_sum)
    
    output_layer_sum = np.dot(hidden_layer_output,output_weights) + output_bias
    predicted_output = sigmoid(output_layer_sum)
    
    # Backpropagation

    error = target - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden = d_predicted_output.dot(output_weights.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)
    
    output_weights += hidden_layer_output.T.dot(d_predicted_output)*learn_rate
    output_bias += np.sum(d_predicted_output,axis=0)*learn_rate
   
    hidden_weights += inputs.T.dot(d_hidden)*learn_rate
    hidden_bias += np.sum(d_hidden,axis=0)*learn_rate
    

            
# Printing the Parameters 
print('Hidden Weights: ')
print(*hidden_weights)
print('Hidden Bias: ')
print(*hidden_bias)
print('Output Weights: ')
print(*output_weights)
print('Output_Bias: ')
print(*output_bias)

print("Predicted Output: ")
print(*predicted_output)

# Error Calculation 
difference = target - predicted_output
difference

# Accuracy Calculation 

accuracy = 0
for i in range(len(difference)):
    accuracy += difference[i][0]

accuracy = (1 + accuracy/len(difference))*100
print("Average Accuracy of predictions: ",accuracy)
        