'''

2018 Assigment One : Differential Evolution
    
Scafolding code

Complete the missing code at the locations marked 
with 'INSERT MISSING CODE HERE'

To run task_2 you will need to download an unzip the file dataset.zip

If you have questions, drop by in one of the pracs on Wednesday 
     11am-1pm in S503 or 3pm-5pm in S517
You can also send questions via email to f.maire@qut.edu.au


'''

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn import model_selection

import random

# ----------------------------------------------------------------------------

def denormalize(valueMin, valueMax, valueToBeDenorm):
    '''
    Function to denormalize a variable
    
    @params
        valueMin: lower bound
        valueMax: upper bound
        valueToBeDenorm: normalized value
    '''
    return valueToBeDenorm * (valueMax - valueMin) + valueMin

def differential_evolution(fobj, 
                           bounds, 
                           mut=2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = True):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    n_dimensions = len(bounds) # dimension of the input space of 'fobj'
    #    This generates our initial population of 10 random vectors. 
    #    Each component x[i] is normalized between [0, 1]. 
    #    We will use the bounds to denormalize each component only for 
    #    evaluating them with fobj.
    
    'INSERT MISSING CODE HERE'
    print('init')
    w = np.random.rand(popsize, n_dimensions)
    '''
    w_denorm = []
    for j in range(0,len(w)):
        w_denorm.append([denormalize(bounds[i][0], bounds[i][1],w[i]) for i in range(0,n_dimensions)])
        '''
    '''
    w = [random.uniform(preprocessing.normalize([[bounds[x][0],bounds[x][1]]])[0][0], preprocessing.normalize([[bounds[x][0],bounds[x][1]]])[0][1]) for x in range(0,n_dimensions)]
    '''
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    w_denorm = min_b + w * diff
    print(w_denorm[0])
    print(len(w))
    cost = np.asarray([fobj(a_w) for a_w in w_denorm])
    best_idx = np.argmin(cost)
    best = w_denorm[best_idx]
    
    
    if verbose:
        print(
        '** Lowest cost in initial population = {} '
        .format(cost[best_idx]))        
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i))        
        mutant = []
        trial = []
        for j in range(0,popsize):
            'INSERT MISSING CODE HERE'
            # Initialize list of indices (range of popsize without the current value of j)
            rand_idx = [an_index for an_index in range(popsize) if an_index != j]
            # pick distict 3 index from the list
            idx_a, idx_b, idx_c = random.sample(rand_idx,3)
            # Initialize a b and c
            a = w[idx_a]
            b = w[idx_b]
            c = w[idx_c]
            
            # append mutant vector and use np.clip to make it in range 0,1
            mutant_content = a + mut * (b-c)
            mutant_content = np.clip(mutant_content, 0, 1)
            
            # init trial vector
            # 1. Generate list of bool that returns true if a random number (range from 0 to 1) is below crossp
            crossps = np.random.rand(n_dimensions) < crossp
            # 2. if the bool is true, append mutant, else, append original weight to trial vector
            trial = np.where(crossps, mutant_content, w[j])
            
            trial_denorm = [denormalize(bounds[i][0], bounds[i][1], trial[i]) for i in range(0,len(trial))]
            
            cost_challenger = fobj(trial_denorm)
            
            #if the cost is lower than the cost in current index, change
            if cost_challenger < cost[j]:
                cost[j] = cost_challenger
                w[j] = trial
                #if the cost is lower than the best in the population, change best index
                if cost_challenger < cost[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, cost[best_idx]

# ----------------------------------------------------------------------------

def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate. 
    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def fmodel(x, w):
        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).
        '''
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            assert type(x) is np.ndarray
            y = np.zeros_like(x)
        #'INSERT MISSING CODE HERE'
            for i in range(0,len(y)):
                for j in range(0,len(w)):
                    if j == 0: 
                        y[i] = w[0]
                    else:
                        y[i] = y[i] + w[j] * (x[i]**j)
        return y

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        Y_pred = fmodel(X, w)
        return np.sqrt(sum((Y - Y_pred)**2)/len(Y)) #'INSERT MISSING CODE HERE'


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        # w : best solution so far
        # c_w : cost of w        
        # Stop when solution cost is less than the target cost
        if c_w< target_cost: #'INSERT MISSING CODE HERE':
            break
        
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
    # result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000))    
    # w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()    
    

# ----------------------------------------------------------------------------

def task_2():
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = clf.score(X_train_transformed, y_train) #'INSERT MISSING CODE HERE'
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed =  scaler.transform(X_test)#'INSERT MISSING CODE HERE'


    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=10, 
            maxiter=20,
            verbose=True)
    
    for i, p in enumerate(de_gen):
        w, c_w = p  #'INSERT MISSING CODE HERE'
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    
# ----------------------------------------------------------------------------

def task_3():
    '''
    Place holder for Task 3    
    '''
    pass
    'INSERT MISSING CODE HERE'


# ----------------------------------------------------------------------------


if __name__ == "__main__":
#    pass
    task_1()    
#    task_2()    
#    task_3()    

