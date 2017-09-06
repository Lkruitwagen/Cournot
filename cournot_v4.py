from scipy import optimize,arange
import sys
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

#vectorized 2P-3T cournot almost good

#vectorize... check!
#add three periods... check!
#xn is now the subgame strategy for player n.
#let's make something 3 turns long... check!
#let's just get 2x3 basic thing like before then we'll add complicated period stuff... check!

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
    #n's profit for strategy xn given full array x
    #...
   
    #pass over x as an array

 
    #print 'profit xn', xn
    #print 'profit xn', xn[0]
    #print 'profit xn', xn[1]
    #print 'profit xn', xn[2]
    #print 'x_allmn',x_allmn
    x = np.insert(x_allmn,n,xn,0)
    #print '\rx',x,
    add=0.0
    #print '\r',x.shape[1],
    for i in range(int(x.shape[1])):
        #print '\rx.T',x.T[i],
        #print 'xn',xn[i]
        add+= price(x.T[i],b)*xn[i]-cost(xn[i],cn)
    #make negative
    return -1.0*add

def reaction(x_allmn,cn,b,n):
	#xn's best response to x_all-n
    #~~~!!!! need to call -profit differently - return a vector instead of a single val. 
    #xn's best response to x_all-n
    #xn = optimize.brute(lambda x: -profit(x,np.insert(x_allmn,n,x),cn,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    xn = optimize.brute(profit, ((0.3,0.4),(0.3,0.4),(0.3,0.4)), args = (x_allmn,n,cn,b), finish = optimize.fmin)

    #print 'rxn xn:', xn
    return xn

def vector_reaction(x,param): # vector param = (b,c1,c2)
    #use np.delete to 'pop' the nth strategy
    rex = np.array([x[0:3],x[3:6]])
    print rex

    #print np.delete(x,0,0)
    reso = np.array(rex) - np.array([reaction(np.delete(rex,0,0),param[1],param[0],0),reaction(np.delete(rex,1,0),param[2],param[0],1)])
    print 'reso', reso
    return reso.flatten()
def main():

	#lets vectorize and do it twice
    param = [1.0,0.0,0.0]
    x0 = np.array([[0.3, 0.3,0.3],[0.3,0.3,0.3]])
    #i.e. x_arr - f(x_arr) = 0
    ans = optimize.fsolve(vector_reaction, x0, args = (param), xtol = 1e-12)
    print 'ans'
    print ans

if __name__ == '__main__':
    main()