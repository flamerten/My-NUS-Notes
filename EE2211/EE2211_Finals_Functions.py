import numpy as np
from numpy.linalg import *
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

import matplotlib.pyplot as plt
import math

#NOTE Code on decision trees and k means clustering are not included


#Classification
def classify_train(data):
    """Uses OneHotEncoding on dataset(input type > list) to create a n x n matri
    where n is the length of the array"""
    enc = OneHotEncoder(sparse = False)
    np_array = np.array(list(map(lambda x: [x],data)))
    return enc.fit_transform(np_array)

def classify_res(numpy_array):
    """Takes the result and finds the max in each row"""
    res = []
    for row in numpy_array:
        res.append(list(map(lambda x: 1 if max(row) == x else 0,row)))
    
    return np.array(res)
    

#Helper Functions
def list_to_np(lis):
    """create a n x 1 matrix given list"""
    return np.array(list(map(lambda x: [x],lis)))

def right_inv(m):
    """for underdetermined system"""
    return m.T@inv(m@m.T)

def left_inv(m):
    """for overdetermined system"""
    return inv(m.T@m)@m.T


def insert_bias(mat):
    """insert bias to matrix on left side"""
    rows,columns = mat.shape
    #offset = np.array( [[1]]* rows )
    #return np.concatenate((offset,mat),axis = 1)

    x0 = np.ones((rows,1))
    return np.hstack((x0,mat))


def unique(mat):
    """remove repeated rows from dataset"""
    return np.unique(mat,axis = 0)

def mse_score(mat1,mat2):
    return np.square(np.subtract(mat1,mat2)).mean()


#Solving
def solve_matrix(x,y, bias = True):
    """returns the least square solution, bias = False -> do not insert bias into features
    No complexity parameter-> non ridge """
    
    if bias:
        new_x = insert_bias(x)
    else:
        new_x = x
    
    if new_x.shape[0] == new_x.shape[1]:
        return inv(new_x)@y
    
    elif new_x.shape[0] > new_x.shape[1]:
        #more rows than columns, over determined
        if matrix_rank(new_x)== matrix_rank(np.concatenate((new_x,y), axis = 1)):
            print("rank(x) = rank([x,y])")
            print("There IS an exact solution")
        return left_inv(new_x)@y
    else:
        if matrix_rank(new_x) < matrix_rank(np.concatenate((new_x,y), axis = 1)):
            print("Warning! No solution as system is inconsistent")
            print("rank(x) < rank([x,y])")
            
        return right_inv(new_x)@y

def predict(X,w, bias = True):
    """bias = False -> do not insert bias, depends on how the solution was created
    X is the solution equation from slve w is the parameters. """
    if bias:
        new_X = insert_bias(X)
    else:
        new_X = X
        
    return new_X@w

def ridge_solve(x,y,bias = True, complex_param  = 0.001, overide = False):
    """Use overide if you would like to manualy select dual ( D ) or primal ( P )"""

    def primal():
        print("Primal Form")
        return inv( new_x.T@new_x +\
                    complex_param*np.identity(new_x.shape[1])) @ new_x.T@y
    def dual():
        print("Dual Form")
        return new_x.T@inv(new_x@new_x.T +\
                           complex_param * np.identity(new_x.shape[0]))@y
        
    if bias:
        new_x = insert_bias(x)
    else:
        new_x = x

    if overide == "P":
        return primal()
    elif overide == "D":
        return dual()
    

    if new_x.shape[0] == new_x.shape[1]:
        print("Warning: Number of Rows = Number of Columns", end = " ")
        print("Dual calculation is performed")
        

    if new_x.shape[0] > new_x.shape[1]:
        return primal()
    else:
        return dual()


#Calculate Node Impurity
def score_node(lis, param = "G"):
    """Use a lis of numbers to represent different classes Metric is:
    G > Gini , E > Entropy, M > Misclassification rate
    Use create_node if you need to :)"""
    p = []
    total = len(lis)
    for item in set(lis):
        p.append(lis.count(item) / total)
    if param == "G":
        return 1 - sum(map(lambda x: x**2,p))
    elif param == "E":
        return - sum(map(lambda x: x * math.log(x,2),p))
    elif param == "M":
        return 1 - max(p)
    else:
        print("Param not recognised")

def score_depth(depth, param = "G"):
    """Use a lis of numbers to represent different classes. Metric is:
    G > Gini , E > Entropy, M > Misclassification rate
    *Depth should b a list of different branches > a list of lists"""

    number = list(map(len,depth))
    total_samples = sum(number)
    fraction = list(map(lambda x: x/total_samples, number))
    
    score_lis = list(map(lambda x: score_node(x,param = param),depth))


    print("Total Scores:",score_lis)
    overall = 0
    for i in range(len(fraction)):
        overall += fraction[i]*score_lis[i]

    print("Overall Score:", overall)
    return

def create_node(tup):
    """Tuple of tuples, ((a,b),(a1,b1)....) Or lists of lists should work as welll
    a is an arbitrary number, b is the number of items"""
    lis = []
    for t in tup:
        lis = lis + [t[0]] * t[1]
    return lis

#convert to polynomial
def polynomial(mat,order):
    """np.array of matrix, order
    NOTE: Do this for both training set AND test set"""
    Poly = PolynomialFeatures(order)
    res =  Poly.fit_transform(mat)
    print(Poly.get_feature_names(), ">>",len(Poly.get_feature_names()))
    return res

#person
def linear_corr(arr):
    """takes in np array and check persons correlations
    Note that it migh be easier to compare 2 values  - matrix of of (n,2)
    Then the 2 numbers that  != 1 are the correlations between the 2 values"""
    print(np.corrcoef(arr,rowvar = False))
    return

#testing array

if __name__ == "__main__":
    x = np.array([[1,2,3],[4,5,6]])
    print("Testing array")
    print(x)
    print("""
Please run this file in your idle and then call functions during the EE2211 exam.
Code on decision trees and K means clustering is not included
General idea, you use np.array to create a matrix X and y and solve it with the solve function.
Bias is automatically included to so remove it with the function parameters if neccesarry
""")
