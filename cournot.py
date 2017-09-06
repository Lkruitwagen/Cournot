from scipy import optimize,arange
from numpy import array
import matplotlib.pyplot as plt
#matplotlib inline

#basic cournot with two firms

def price(x,b):
	#x is an array [x1, x2]
    return 1-x[0]-b*x[1]

def cost(x,c):
    if x == 0:
     cost = 0
    else:
     cost = c*x
    return cost

def profit(x1,x2,c1,b):
    #x1's profit given x1 and x2
    #...
    #try to vectorize first
    x = array([x1,x2])
    #pass over x as an array
    return price(x,b)*x1-cost(x1,c1)

def reaction(x2,c1,b):
    x1 = optimize.brute(lambda x: -profit(x,x2,c1,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    return x1[0]

def vector_reaction(x,param): # vector param = (b,c1,c2)
    return array(x)-array([reaction(x[1],param[1],param[0]),reaction(x[0],param[2],param[0])])

def main():

	#lets vectorize and do it twice
    param = [1.0,0.0,0.0]
    x0 = [0.3, 0.3]
    #i.e. x_arr - f(x_arr) = 0
    ans = optimize.fsolve(vector_reaction, x0, args = (param))
    print ans

if __name__ == '__main__':
    main()