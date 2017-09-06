from scipy import optimize,arange
from math import *
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

#vectorised 2P-3T cournot now using basinhopping

#1. get in the data, etc. ... CHECK!
#2. make arbitrary to nxm ... CHECK!
#3. make investment game ... v7
#4. div;expl

#vectorize... check!
#add three periods... check!
#xn is now the subgame strategy for player n.
#let's make something 3 turns long... check!
#let's just get 2x3 basic thing like before then we'll add complicated period stuff... check!


def read_csv_dict(filename):
  out=[]
  dic={}
  with open(filename,'rb') as infile:
    reader = csv.reader(infile)
    headers = next(reader)
    #print headers
    for row in reader:
      dic = {key: value for key, value in zip(headers, row)}
      out.append(dic)
  #print 'years'
  infile.close()

  return out,headers

def price(x,b):
	#x is an array [x1, x2]
    return 1-sum(x)

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

def objective(xn,x_allmn,n,cn,b):
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

def oil_price(curve,dem):
    sup_low = float(curve[0])
    sup_med = float(curve[0])+float(curve[1])
    sup_hig = sup_med+float(curve[2])
    #print sup_low,sup_med,sup_hig
    y1 = 20.0*(dem/sup_low)*(1/(1+exp(-10.0*dem)))*(1-(1/(1+exp(-10.0*(dem-sup_low)))))
    y2 = (20.0+(50.0-20.0)*((dem-sup_low)/(sup_med-sup_low)))*(1/(1+exp(-10.0*(dem-sup_low))))*(1-(1/(1+exp(-10.0*(dem-sup_med)))))
    #print y2
    y3 = (50.0+(80.0-50.0)*((dem-sup_med)/(sup_hig-sup_med)))*((1/(1+exp(-10.0*(dem-sup_med)))))*(1-(1/(1+exp(-1.0*(dem-sup_hig))))) #*(1/(1+exp(-10.0*(dem-sup_med))))*(1-(1/(1+exp(-10.0*(dem-sup_hig)))))
    y4 = (80.0+(140.0-80.0)*((dem-sup_hig)/(sup_hig*0.2)))*((1/(1+exp(-1.0*(dem-sup_hig)))))*(1-(1/(1+exp(-1.0*(dem-sup_hig*1.2))))) #*(1/(1+exp(-10.0*(dem-sup_hig))))*(1-(1/(1+exp(-10.0*(dem-(sup_hig*1.2))))))
    y5 = (140.0)*((1/(1+exp(-1.0*(dem-sup_hig*1.2))))) #*(1/(1+exp(-10.0*(dem-sup_hig))))*(1-(1/(1+exp(-10.0*(dem-(sup_hig*1.2))))))
    
    #print 'y1',y1
    #print 'y2',y2
    #print 'y3',y3
    #print 'y4',y4
    #print 'y5', y5
    y = y1+y2+y3+y4+y5
    #print y
  
    #print 'price'
    return y
  

def reaction(x_allmn,cn,b,n,n_cos,n_years):
	#xn's best response to x_all-n
    #~~~!!!! need to call -profit differently - return a vector instead of a single val. 
    #xn's best response to x_all-n
    #xn = optimize.brute(lambda x: -profit(x,np.insert(x_allmn,n,x),cn,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    #xn = optimize.brute(profit, ((0.3,0.4),(0.3,0.4),(0.3,0.4)), args = (x_allmn,n,cn,b), finish = optimize.fmin)
    b = (0.0,1.0)
    bnds = tuple([b for j in range(n_years)])

    minimizer_kwargs = {'args': (x_allmn,n,cn,b), 'bounds':bnds}
    x0 = np.full((1,n_years),0.3)
    xn = optimize.basinhopping(profit, x0, minimizer_kwargs=minimizer_kwargs) #, niter=200)

    #print 'rxn xn:', xn
    print 'xn.x',xn.x, type(xn.x)

    return xn.x

def vector_reaction(x,param): # vector param = (b,c1,c2)
    #use np.delete to 'pop' the nth strategy
    rex = np.reshape(x,(param['n_cos'],param['n_years']))
    print rex

    #print np.delete(x,0,0)
    chex = []
    for i in range(param['n_cos']):
        chex.append(reaction(np.delete(rex,i,0),param['consts'][i],param['consts'][-1],i,param['n_cos'],param['n_years']))
    #print chex
    reso = np.array(rex) - np.array(chex)
    print 'reso', sum(sum(np.absolute(reso)))
    return reso.flatten()

def main():

    #import our data
    companies, comp_h = read_csv_dict('5p.csv')
    print comp_h
    n_companies = len(companies)
    print 'n_comp=',n_companies
    scenario, scen_h = read_csv_dict('SCENDATA_E3G-ERSv2_20170510_2012.csv')
    print scen_h
    n_years = len(scenario)
    print 'n_years',n_years

    print oil_price([35,25,35],140.0)
    print oil_price([35,25,35],100.0)
    print oil_price([35,25,35],80.0)
    print oil_price([35,25,35],75.0)
    print oil_price([35,25,35],55.0)
    print oil_price([35,25,35],25.0)



    # but okay lets not do that many yet
    n_companies = 2
    n_years = 3
    x0 = np.full((n_companies,n_years),0.3)
    consts = [0.0 for j in range(n_companies)]+[1.0]
    print 'consts',consts

    param = {'consts': consts, 'n_cos': n_companies, 'n_years':n_years}
    #x0 = np.array([[0.3, 0.3,0.3],[0.3,0.3,0.3]])
    #i.e. x_arr - f(x_arr) = 0
    ans = optimize.fsolve(vector_reaction, x0, args = (param), xtol = 1e-8)
    print 'ans'
    print ans

if __name__ == '__main__':
    main()