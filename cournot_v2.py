from scipy import optimize,arange
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline


#basic cournot vectorised
#vectorize... check?
#add second period...

def price(x,b):
	#x is an array [x1, x2]
    return 1-x[0]-b*x[1]

def cost(x,c):
    if x == 0:
     cost = 0
    else:
     cost = c*x
    return cost

def profit(xn,x_allmn,n,cn,b):
    #xn's profit given the array x
    #...
   
    #pass over x as an array
    x = np.insert(x_allmn,n,xn,0)
    print 'x=', x
    return -1.0*(price(x,b)*xn-cost(xn,cn))

def reaction(x_allmn,cn,b,n):
	#xn's best response to x_all-n
    #xn = optimize.brute(lambda x: -profit(x,np.insert(x_allmn,n,x),cn,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    xn = optimize.brute(profit, ((0,1),), args = (x_allmn,n,cn,b))

    print 'xn=',xn
    return xn[0]

def vector_reaction(x,param): # vector param = (b,c1,c2)
    #use np.delete to 'pop' the nth strategy
    return np.array(x)-np.array([reaction(np.delete(x,0),param[1],param[0],0),reaction(np.delete(x,1),param[2],param[0],1)])

def main():

	#lets vectorize and do it twice
    param = [1.0,0.0,0.0]
    x0 = [0.3, 0.3]
    #i.e. x_arr - f(x_arr) = 0
    ans = optimize.fsolve(vector_reaction, x0, args = (param))
    print ans

if __name__ == '__main__':
    main()