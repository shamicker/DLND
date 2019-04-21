
# **Deep Learning**
inside the field of machine learning, using massive datasets, accelerated computing on GPU (graphics processing units)

## **PyTorch**
- open-source Python framework
- from Facebook's AI Research team (FAIR)
- used for developing deep neural networks

## **Neural Network**

- imitates how a brain works
- in the same way as in a brain
    - the *input* nodes are one-way streets to a calculation node
    - some decision happens there, and produces a one-way action to the output
        - in a brain it happens or it doesn't, but here it should always happen but with different results


- basically, it finds a boundary between different categories (like flower species)

## General deep learning terms

### **Features**
- inputs
- the data that is input into a neural network for it to study
- `features.shape` gives `n_records`, `n_inputs`
    - `n_records` is the number of, say, students in a class
    - `n_inputs` is the number of inputs per student

### **Targets** 
- the desired outcomes, usually already known in order to train a model
- denoted by $y$

### **Prediction** 
- the output; the prediction of how items are categorized
- aka **score**
- aka **logits**
- denoted by $\hat{y}$
- basically the weights times the features
    - $\hat{y} = Wx$ or $\hat{y} = f(Wx)$
    - often put through an activation function, but not always (depends what you need the output to be)
    - for any specific weight located at $i$: $\hat{y} = f(\sum_i w_{i}x_{i})$ 


### **Activation Functions**
- a way of presenting output; preparing it for the next step

### **Linear Boundaries**

- a boundary that's linear
- this is the simplest solution/output produced by a neural network (ie to differentiate between the target categories)
- calculated with $Wx + b = 0$, where `W` is some weights, and `b` is a bias


- there are higher dimensions: curves, planes, etc.

### **Non-Linear Models**

![combining linear regions](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/combining_regions.PNG)

### **Perceptron**
- a building block of neural networks
- a visualisation of an equation into a small graph, where
    - each input $x$ is in a node
    - the weights $w$ are labeled on the *edges* (the arrows) of the input nodes
    - the bias $b$ can either be in the calculation node or viewed as an *input* node
        - usually the bias is considered a *weight* for an input of constant $1$, as in this image
    - the calculation is also in a node
    - the prediction is the return value

![perceptron with summing equation](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-summation.PNG)

#### **Examples of Perceptrons as Logical Operators**

An AND perceptron:
![an AND perceptron visualized](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/and-perceptron.PNG)

An OR perceptron:
![an OR perceptron visualized](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-or.PNG)


An XOR perceptron, which is either or, but not both and not neither:
![an XOR perceptron visualized](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-xor.PNG)


**2 different visuals for the XOR perceptron**
![an XOR neural network](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/xor-neuralnetwork.PNG)

![an XOR multi-layer perceptron](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/xor-multi-layer-perceptron.PNG)

#### **Perceptron Algorithm**
- adjusting the linear boundary by "asking" each data point if it's been classified correctly or incorrectly



### **Logistic Regression**
- "the building block of all that constitutes Deep Learning!"
- I think this is the estimation of the relationship between variables (say, x and y in a scatterplot)
- basically:
    - take your data
    - pick a random model
    - calculate the error
    - minimize the error and obtain a better model
    - enjoy!

The term "regression" goes back to the 19th century and was used to describe a biological phenomenon where the heights of a tall person's descendants *regressed* down towards a normal average.


### **Gradient Descent**
- the negative slope of the output (for single layer) or loss function (for multi layer) - I think??
    - negative because the derivative returns the steepest ascent, but we want descent
- is the resulting vector of partial derivatives (for multi layer), with respect to the weights (since the error fn is of the weights)
    - points in the direction of fastest change (steepest ascent)
- 


### **Learning Rate**
- when descending via gradient descent, we don't want to take too large of a step so we scale it down. This scaling (amount) is the learning rate.

### **Multi-layer Networks**
- first layer is the **input layer**
- inside layer(s) are the **hidden layer**
- final layer is the **output layer**

For multi-class, multi-layer networks, usually have an ouput node for each class, giving a probability for each class.

### **Feedforward**
- the forward step in training; from in to out
- the weights of each layer are denoted by the superscript $(i)$, starting at $1$ for the input layer
    - ie $W^{(1)}$, $W^{(i)}$

- the whole equation, if using Sigmoid as the activation functions:
$\hat{y} = \sigma \big(W^{(2)}\big) \cdot \sigma \big(W^{(1)}\big) \cdot x$

### **Backpropagation**
- asks "Which model within the hidden layer is doing better?"
- asks each point to give a grade, and then it updates the weights of that layer 
- Think of this as flipping the model around
    - *You're now doing the same thing in the reverse direction, but **using the Error as inputs**.*

Steps:
- Do a feedforward operation 
- Calculate the error
- Optionally, calculate the **error term**: the error times the slope of the prediction
    - "optionally" - I believe this is calculated when calculating manually. It's sort of built in to the modules
- Run the feedforward operation backwards (which is backpropagation) to spread the error to each of the weights
- Use this to update the weights, and get a better model
- Repeat until we have a good model


So in code, or in math terms:
- forward step
    - $\hat{y} = f(W\cdot x)$
- error
    - $E = y - \hat{y}$
- error term
    - $\delta = E * \sigma'(x)$
    - $\delta = (y - \hat{y}) * f'(\sum_i w_{i}x_{i})$
- backprop
    - $\Delta w_{i} = \Delta w_{i} + \delta x_{i}$  
- update the weights
    - $W = W + learnrate * \Delta w / len(W)$  
    -- This seems to be inaccurate. I think it's without the $learnrate$

#### Backpropagation Math

![very simple network]("./backprop_network.png")

In case the image doesn't appear:
- 2 input/feature nodes: $x = [0.1, 0.3]$
- the 2 weights: $W^{(1)} = [0.4, -0.2]$
- there's a single hidden layer with a single node  
- activation function of hidden layer ($h$) is $sigmoid()$
- weight of hidden layer to the single output node is $[0.1]$
- activation function of output layer is $sigmoid()$
- as usual, the output is $\hat{y}$

Start with the forward pass.  
- Calculate the input to the hidden layer  
$h = features \times weights$  
$h = W^{(1)}x$  
$h = \sum_{ij} w_{ij}x_{ij}$  
$h = 0.1\times 0.4 + (-0.2) \times 0.3 = -0.02$

- Calculate the output of the hidden unit ($h$, activated)  
$a = \sigma(h)$  
$a =\sigma(-0.02)$  
$a = 0.495$

- Calculate the input of the output layer (hidden output $a$, times weights, activated)   
$\hat{y} = \sigma(W^{(2)}\cdot a)$  
$\hat{y} = \sigma(0.1\times 0.495)$  
$\hat{y} = 0.512$

Now for the backpropagation.  
We already know the derivative of the sigmoid function is $\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$, and that the error $E = y - \hat{y}$. We also will have a `learning_rate` of $\eta = 0.5$.

Calculate the **error terms**.

- Calculate the **output layer error** (error times output's derivative)  
$\delta^{o} = E \cdot \hat{y}'$  
$\delta^{o} = (y - \hat{y})\cdot \hat{y}'$  
$\delta^{o} = (y - \hat{y})\cdot\hat{y}\cdot(1 - \hat{y})$  
(aka)  
$\delta^{o} = (y - \hat{y})\cdot\sigma'(W^{(2)}\cdot a)$  
$\delta^{o} = (y - \hat{y})\cdot \sigma(W^{(2)}\cdot a)\cdot(1 - \sigma(W^{(2)}\cdot a))$   
$\delta^{o} = (1 - 0.512) \times 0.512 \times (1 - 0.512)$  
$\delta^{o} = 0.122$

- Calculate the **hidden layer error**.  
$\delta^{h} = W^{(2)}\cdot E\cdot \hat{y}'\cdot a'$  
$\delta^{h} = W^{(2)}\cdot\delta^{o}\cdot\sigma'(h)$  
$\delta^{h} = W^{(2)}\cdot\delta^{o}\cdot\sigma(h)\cdot(1 - \sigma(h))$  
$\delta^{h} = 0.1 \times0.122 \times 0.495 \times(1 - 0.495)$  
$\delta^{h} = 0.003$  

Calculate gradient descent steps.

- Calculate the weight step from the hidden to output layer. (learning rate times output layer error, times the hidden layer activation value)  
$\Delta W^{(2)} = \eta\cdot\delta^{o}\cdot a$  
$\Delta W^{(2)} = 0.5 \times 0.122 \times 0.495$  
$\Delta W^{(2)} = 0.0302$  

- Calculate the weight step from the input to hidden layer $w_{i}$ (learning rate times hidden layer error, times the input values)  
$\delta w_{i} = \eta\cdot\delta^{h}\cdot x_{i}$  
$\delta w_{1} = 0.5\times 0.003\times 0.1$  
$\delta w_{2} = 0.5\times 0.003\times 0.3$  
$\delta w_{1} = 0.00015$  
$\delta w_{2} = 0.00045$  
$\delta w = (0.00015, 0.00045)$

## Problems that will come up
While we can just bulldoze ahead and train a neural network, there are some issues that will appear pretty soon.

Here are some of them, with some popular solutions.

### Vanishing Gradient
The max derivative of the sigmoid function is 0.25 (think about it!), so
- errors in the output layer get reduced by at least 75%, and
- errors in the hidden layer are scaled down by at least 93.75%!!

So if there are lots of layers, using a sigmoid activation function produces itty, bitty, teeny, tiny steps knows as the **vanishing gradient**.

Solutions: use other activation functions.

### Over- and Under-fitting
- We'll err on the side of overfitting, and then use techniques to make it fit better (like, a belt to hold up too-big pants, rather than using the too-small pants)



#### Early Stopping
- just train for less epochs
    - Save your model at points where both the training and the testing losses diminish.

#### Regularization
Since our activation function creates a *vanishing gradient* (I think this is the reason?), we will *regularize* the data by penalizing large weights.
- example: for points $(1, 1)$ and $(-1, -1)$  
    ($10x_{1} + 10x_{2}$) has an error of $0.000000012$  
    ($x_{1} + x_{2}$) has an error of $0.12$  
    So it's most likely that the former is over-fitted
    
In the above example, we penalize the large weights (10x) because they produce less error.

There are a couple ways of doing this.
    

##### L1 Regularization
Add the sum of absolute weights to the error function

$E + \lambda(|w_{1}| + \cdots + |w_{n}|)$  

The $\lambda$ is just a coefficient of scale

When applied, tend to end up with sparse vectors.  
The small weights tend to go to zero, so we're basically only left with the large weights.
- good for feature selection

##### L2 Regularization
Add the sum of squared weights to the error function  
$E + \lambda(w_{1}^2 + \cdots + w_{n}^2)$

Again, $\lambda$ is a coefficient of scale

When applied, tends to try to keep all weights small-ish.  
- generally better for training models

#### Dropout
- turn off a percentage of nodes (on average) during training
- purpose is to not rely on specific nodes too much

### Local Minima
Sometimes we find ourselves in a local valley, which is the lowest point in sight but isn't the bottom of the mountain (aka, isn't the best solution)

Some ways to get around this:

#### Random Restart
- start at a variety of places (restart in diff places). This increases the chance of one of the solutions being the "bottom of the mountain"

#### Momentum
- The idea is to take previous steps into consideration when choosing the direction of the steps.
- weigh each previous step - the last step by a lot, preceding steps less and less
    - using $\beta$ as a constant between $0$ and $1$
    - the previous step is multiplied by $1$, the step before that by $\beta$, etc
    
    
$$step_{(n)} \rightarrow 1step_{(n)} + \beta step_{(n-1)} + \beta^2 step_{(n-2)} + \cdots$$

### Time and Memory Usage
If we use all our data in a single training session, it takes *forever* and we've used up all the data so we can't train anymore. What if we want to train more?

#### Stochastic Gradient Descent (SGD)
Instead of using all the data in one go, break up the data into batches

This results in several smaller, less accurate steps, but you can guage your progress, and it takes less time.

### Learning Rate Decay
- if your learning rate is too big, you could just bounce around the local minimum
- if it's too small, you could be taking forever (steps are too small)

The best learning rates are those that start out with big steps, and take smaller steps as time goes on; as you approach the minimum

## **One-Hot Encoding**
- also called getting dummies
- the act of binary-izing multiple classes, or multiple outcomes
    - in math terms, it's getting an **identity matrix**
    - it's the matrix equivalent of $1$:
        - you can multiply ANY matrix by an identity matrix and you'll get the original matrix back!
        ![it's an identity matrix](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/Identity_matrix.PNG)
        
![one-hot encoding](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/one-hot_encoding.PNG)

## **Activation Functions**
- a way of presenting output; preparing it for the next step

### **Step Function**
- the yes/no kind
- for binary predictions
- $Wx + b = 0$
    - whether a data point is above or below the *linear boundary*
    - aka if the data point output is positive or negative

### **Sigmoid Function**
- probabilities of the outputs to be above or below the linear boundary (or correctly classified)
- the high end approaches 1, the negative approaches 0, the middle is 50% (right at the linear boundary)
- this can be for binary predictions or multi-layer networks
- denoted as $$\sigma(x) = \frac{1}{(1 + e^{-x})}$$

### **Softmax Function**
- for multi-class networks
- this "squishes" the output, or **normalizes** it into a probability, where all the results must add up to 1
- this also eliminates the problem of negative outputs (because $e^{negative-number}$ is positive)
- the formula is $e^{each-output}/\Sigma e^{each-output}$  
![probability of class i](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/softmax_function.PNG)

### **More Activation Functions**
- **TanH**
- **ReLU** is the most popular, simplest, and apparently extremely effective!
![more activation functions](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/more_activation_functions.png)

## **Error Functions**
- a function of the weights
- tell us how far we are from the solution; the distance from the goal
- aka **loss**
- aka **cost**

There are different kinds of error functions:
- discrete outputs (yes/no)
- continuous outputs (probabilities)

and different formulas for error functions:
- log-loss (for 2 possible outputs)
- cross entropy (for 3+ possible outputs)
    - sum of the negative natural logs of the probabilities
    

### **Log Loss**/ **Cross Entropy**
- like everything in neural networks (I think?) we usually calculate with natural logs ($ln$), or base e
- done to get all-positive numbers
- we take the negative of the log of the probabilities
- log loss is for 2 classes, cross entropy is for multi-classes


The log of anything between $0$ and $1$ is a negative number, and the log of $1$ is $0$.

We know that a high probability of success approaches 1 (100%).
When we take the negative log of that probability, we get a low number. So think of this *negative log* as the **errors**.

Because for each input, the outcome is either "yes" or "no"......?????

* Best used for classification problems.


## Calculating Loss
- various ways to calculate loss, depending on where you're heading

### **Mean Squared Loss**
- I believe this is kind of like the *variance*
    - taking the difference of the targets and outputs, squaring the difference, and then calculating this mean
- often used in regression and binary classification problems  
![mean squared loss equation](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/mean_squared_loss_equation.jpg)

## **Multi-Layer Perceptrons vs. Convolutional Neural Networks**  
In image data, it's relevant to know where pixels are in relation to other pixels.
Does every hidden node need to be connected to every original pixel? There may be some redundancy:

### **MLPs**
- for a small (28x28), grayscale image such as in MNIST, already there are 50,000+ parameters!
- must flatten images for inputting
    - 2D spatial info is gone
        - no idea of pixels with reference to other pixels
- uses only **fully-** or **densely-** connected layers
    - ie. all inputs from each layer feed into all nodes in the next layer (or output)

### **CNNs**
- sparsely connected layers
    - 2D spatial relationships are maintained
- accept matrices as input!
    - no flattening
- uses **locally**-connected layers
    - hidden nodes only assess a certain region of the original image
    - less prone to overfitting
- each hidden node works with the same weights ("share a common group of weights", she says) as the others
    - ie. different areas share the same kind of info
        - any relevant pattern could be found in any area of the image
            - ie. the image could be found anywhere in the image

## **Convolutional Neural Networks (CNN)**

**Convolutional Layer:** applies different filters to an input image  
**Convolutional Kernels:** are what these filters are called  
- resulting filtered images will be different from each other
    - each image will have filtered out different information
        - ie. contoured lines (reacting to alternating patterns of dark/light pixels), colours



```python

```
