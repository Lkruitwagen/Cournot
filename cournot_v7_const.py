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
#3. make investment game ... CHECK?
#4. div;expl

#vectorize... check!
#add three periods... check!
#xn is now the subgame strategy for player n.
#let's make something 3 turns long... check!
#let's just get 2x3 basic thing like before then we'll add complicated period stuff... check!
#2017.07.10 - now up to n_cos x n_years x n_quantities cournot optimisation with the given price function. NICE!


cap_exp_low = 5.6   #Exploration efficiency: $/BBL-low
cap_exp_med = 14    #Exploration efficiency: $/BBL-med
cap_exp_hig = 22.4   #Explotation efficiency: $/BBL-hig
cap_upg_low = 5.6    #upgrade efficiency: $/BBL-low
cap_upg_med = 14    #upgrade efficiency: $/BBL-med
cap_upg_hig = 22.4    #upgrade efficiency: $/BBL-hig
cap_pro_low = 8.8   #production efficiency: $/BBL-low
cap_pro_med = 22    #production efficiency: $/BBL-med
cap_pro_hig = 35.2  #production efficiency: $/BBL-hig
cap_exp_gas = 1.62  #Explotation efficiency: $/kCF
cap_upg_gas = 1.62  #upgrade efficiency: $/kCF
cap_pro_gas = 2.54  #production efficiency: $/kCF
leverage_max =  2
por_low = 1.0/6
por_med = 2.0/6
por_hig = 3.0/6  


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


def prod_oil(curve_low, curve_med, curve_hig, dem):
    lift_low = 8.00
    lift_med = 22.00
    lift_hig = 40.00
    #print 'hi'
    P = oil_price([sum(curve_low),sum(curve_med),sum(curve_hig)],dem)
    if (dem-sum(curve_low)-sum(curve_med))>=sum(curve_hig):
        prod_low = curve_low
        prod_med = curve_med
        prod_hig = curve_hig
    elif (dem-sum(curve_low))>=sum(curve_med):
        #in the high area
        prod_low = curve_low
        prod_med = curve_med
        prod_hig = [(dem-sum(curve_low)-sum(curve_med))*float(el)/sum(curve_hig) for el in curve_hig]

    elif (dem)>=sum(curve_low):
        #in the medium area
        prod_low = curve_low
        prod_med = [(dem-sum(curve_low))*float(el)/sum(curve_hig) for el in curve_med]
        prod_hig = [0 for el in curve_hig]

    else:
        #in the low area
        prod_low = [(dem)*float(el)/sum(curve_hig) for el in curve_low]
        prod_med = [0 for el in curve_med]
        prod_hig = [0 for el in curve_hig]
    
    cash_gen = [(P-cap_pro_low)*prod_low[j]+(P-cap_pro_med)*prod_med[j]+(P-cap_pro_hig)*prod_hig[j] for j in range(len(curve_low))]
     #MINUS LIFTING COSTS

    return P, prod_low, prod_med, prod_hig, cash_gen

def print_assets(assets):
    n_cos = len(assets)
    n_years = len(assets[0])
    n_assets = len(assets[0][0])

    for co in range(n_cos):
        print 'company number {}'.format(co)
        print '{} {} {} {} {} {} {} {}'.format('year','cash','UndL','UndM','UndH','DevL','DevM','DevH')
        for y in range(n_years):
            print '{} {} {} {} {} {} {} {}'.format(y, assets[co][y][0],assets[co][y][1],assets[co][y][2],assets[co][y][3],assets[co][y][4],assets[co][y][5],assets[co][y][6])


def state2assets(state,exp_delay,upg_delay,ini_assets,dem):
    #takes a state of allocations and converts it to table of assets

    #allocations: div, exp, upgl, upgm, upgh
    n_cos = int(state.shape[0])
    n_years = int(state.shape[1])
    n_allocs = int(state.shape[2])

    assets = np.zeros((n_cos,n_years,7))
    #cash, undL, undM, undH, devL, devM, devH
    meta = []
    

    for y in range(n_years):
        #print_assets(assets)
        #print 'len',len(assets)
        #print y
        # do price and production
    
        if y==0:
            #do thing with init
            #get oil pice
            curve_low = [el[4] for el in ini_assets]
            curve_med = [el[5] for el in ini_assets]
            curve_high = [el[6] for el in ini_assets]

            P, prod_low, prod_med, prod_hig, cash_gen = prod_oil(curve_low, curve_med, curve_high, dem)
            #print 'P:', P, 'cash_gen:', cash_gen
            meta.append({'P':P, 'cash': cash_gen, 'div':[0 for i in range(n_cos)]})

            for co in range(n_cos):
                # do cash [0]
                assets[co][y][0] = cash_gen[co]

                # do devL[4]
                assets[co][y][4] = ini_assets[co][4] - prod_low[co]

                # do devM[5]
                assets[co][y][5] = ini_assets[co][5] - prod_med[co]

                #do devH[6]
                assets[co][y][6] = ini_assets[co][6] - prod_hig[co]

                #do undL
                assets[co][y][1] = ini_assets[co][1]

                #do undM
                assets[co][y][2] = ini_assets[co][2]

                #do undH
                assets[co][y][3] = ini_assets[co][3]
    
        elif y>0:
            #do stuff with y-1
            curve_low = [el[y-1][4] for el in assets]
            #print 'curve_low', [el[y-1][4] for el in assets]
            curve_med = [el[y-1][4] for el in assets]
            curve_high = [el[y-1][4] for el in assets]

            P, prod_low, prod_med, prod_hig, cash_gen = prod_oil(curve_low, curve_med, curve_high, dem)
            meta.append({'P':P, 'cash': cash_gen, 'div':[assets[co][y-1][0]*state[co][y-1][0] for co in range(n_cos)]})
            
    
        #then do assets in cos:
            for co in range(n_cos):
                #herro
                #error catching: use if statements to limit>=0
                #Add upg delay
                # do cash [0]
                #print 'sum', sum(state[co][y-1])
                assets[co][y][0] = assets[co][y-1][0]*(1.0-sum(state[co][y-1])) + cash_gen[co]

                # do devL[4]
                assets[co][y][4] = assets[co][y-1][4] - prod_low[co] + assets[co][y-1][0]*state[co][y-1][2]/cap_upg_low

                # do devM[5]
                assets[co][y][5] = assets[co][y-1][5] - prod_med[co] + assets[co][y-1][0]*state[co][y-1][3]/cap_upg_med

                #do devH[6]
                assets[co][y][6] = assets[co][y-1][6] - prod_hig[co] + assets[co][y-1][0]*state[co][y-1][4]/cap_upg_hig

                exp_bbl = assets[co][y-1][0]*state[co][y-1][1]/(por_low*cap_exp_low+por_med+cap_exp_med+por_hig*cap_exp_hig)
                
                #do undL
                assets[co][y][1] = assets[co][y-1][1] - assets[co][y-1][0]*state[co][y-1][2]/cap_upg_low + exp_bbl*por_low

                #do undM
                assets[co][y][2] = assets[co][y-1][2] - assets[co][y-1][0]*state[co][y-1][3]/cap_upg_med + exp_bbl*por_med

                #do undH
                assets[co][y][3] = assets[co][y-1][3] - assets[co][y-1][0]*state[co][y-1][4]/cap_upg_hig + exp_bbl*por_hig
    #print_assets(assets)

    y = n_years
    curve_low = [el[y-1][4] for el in assets]
            #print 'curve_low', [el[y-1][4] for el in assets]
    curve_med = [el[y-1][4] for el in assets]
    curve_high = [el[y-1][4] for el in assets]

    P, prod_low, prod_med, prod_hig, cash_gen = prod_oil(curve_low, curve_med, curve_high, dem)
    meta.append({'P':P, 'cash': cash_gen, 'div':[assets[co][y-1][0]*state[co][y-1][0] for co in range(n_cos)]})
    return assets, meta

def objective(xn,x_allmn,n,cn,b, ini_assets):
    #n's profit for strategy xn given full array x
    #...
   
    #pass over x as an array


    #print 'profit xn', xn
    #print 'profit xn', xn[0]
    #print 'profit xn', xn[1]
    #print 'profit xn', xn[2]
    #print 'x_allmn',x_allmn
    #print 'xn', xn
    n_years = x_allmn.shape[1]
    xn = xn.reshape(1,n_years,2) #add a proper len variable
    x = np.insert(x_allmn,n,xn,0)
    #print '\rx',x,
    #assets, meta = state2assets(x,1,1,ini_assets,100)
    #div = sum([el['div'][n] for el in meta])



    EV=0.0
    for yr in range(n_years):
      sumL = np.sum(x,axis=0)[yr][0]
      sumM = np.sum(x,axis=0)[yr][1]

      #EV+=P(50.0,sumL,sumM)*(x[n][yr][0]+x[n][yr][1])-20.0*x[n][yr][0]-50.0*x[n][yr][1]
      if sumL>50.0:
        proL = 50.0*x[n][yr][0]/sumL
      else:
        proL = x[n][yr][0]

      if sumM>(50.0-sumL):
        proM = (50.0-sumM)*x[n][yr][1]/sumM
      else:
        proM = x[n][yr][1]


      EV+=P(50.0,sumL,sumM)*(proL+proM)-20.0*proL-50.0*proM
    return -1.0*EV


def P(d,sumL,sumM):
  A = [20,30,30]
  k = -0.2
  x0 = [
  sumL/2.0,
  sumL+sumM/2.0,
  (sumL+sumM)*1.1
  ]

  y0 = A[0]/(1.0+exp(k*(d-x0[0])))
  y1 = A[1]/(1.0+exp(k*(d-x0[1])))
  y2 = A[2]/(1.0+exp(k*(d-x0[2])))
  #y3 = A[3]/(1.0+exp(k*(d-x0[3])))
  y = y0+y1+y2  #+y3
  #y = y0+y1
  #print y,'\r',
  return y

def con_L(x,n):
    #print 'n',n
    #print 'x',x
    #x-1 > 0 => x>1
    #x+1 >0 => x>-1
    #-x -1 >0 => x<-1
    #-x+1>0 => x<1
    x = x.reshape(3,2)
    #print 'x', x
    #print 'n',n
    #print 'ret', x[0][n],x[1][n],x[2][n],30
    #print -x[0][n]-x[1][n]-x[2][n]+30.0
    return -x[0][0]-x[1][0]-x[2][0]+10.0
    #return x[0][n]+x[1][n]+x[2][n]-40.0
    #return -x[0][n]-x[1][n]-x[2][n]+20.0

def con_M1(x,n):
    x = x.reshape(3,2)
    return x[0][1]-x[0][0]/2.0

def con_M2(x,n):
    x = x.reshape(3,2)
    return x[1][1]-x[1][0]/2.0

def con_M3(x,n):
    x = x.reshape(3,2)
    return x[2][1]-x[2][0]/2.0




def con_bnd(x,n):
    return x[n]

def reaction(x_allmn,cn,b,n,n_cos,n_years, ini_assets):
	#xn's best response to x_all-n
    #~~~!!!! need to call -profit differently - return a vector instead of a single val. 
    #xn's best response to x_all-n
    #xn = optimize.brute(lambda x: -profit(x,np.insert(x_allmn,n,x),cn,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    
    b = (0.0,100.0)
    bnds = tuple([b for j in range(n_years*2)])
    #print 'bnds',bnds
    cons = ([])
    #for i in range(2):
        #neither low or med can >30
    #    cons.append({'type': 'ineq', 'fun': con_L,'args':(i,)})
    cons.append({'type': 'ineq', 'fun': con_L,'args':(0,)})
    cons.append({'type': 'ineq', 'fun': con_M1,'args':(0,)})
    cons.append({'type': 'ineq', 'fun': con_M2,'args':(0,)})
    cons.append({'type': 'ineq', 'fun': con_M3,'args':(0,)})
    
    for i in range(n_years*2):
        cons.append({'type': 'ineq', 'fun': con_bnd,'args':(i,)})
    #print cons
    #raw_input('-->')

    minimizer_kwargs = {'args': (x_allmn,n,cn,b,ini_assets), 'constraints':cons, 'bounds': bnds}
    x0 = np.full((1,n_years,2),10)
    #xn = optimize.brute(objective, ((0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)), args = (x_allmn,n,cn,b, ini_assets), finish = optimize.fmin)
    xn = optimize.minimize(objective, x0, args = (x_allmn,n,cn,b,ini_assets), constraints = cons, bounds = bnds) 
    #xn = optimize.basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs) #, niter=200)

    #print 'rxn xn:', xn
    #print 'xn.x',xn.x, type(xn.x)

    return xn.x

def vector_reaction(x,param): # vector param = (b,c1,c2)
    #use np.delete to 'pop' the nth strategy
    rex = np.reshape(x,(param['n_cos'],param['n_years'],2))
    print 'rex',rex

    #print np.delete(x,0,0)
    chex = []
    for i in range(param['n_cos']):
        temp = reaction(np.delete(rex,i,0),param['consts'][i],param['consts'][-1],i,param['n_cos'],param['n_years'], param['ini_assets'])
        temp = temp.reshape(param['n_years'],2)
        chex.append(temp)
    print 'chex',np.array(chex)

    reso = np.array(rex) - np.array(chex)
    resid = sum(sum(sum(np.absolute(reso))))
    print 'reso all', reso
    print 'resid', resid
    return reso.flatten()

def vector_reaction_o(x,param): # vector param = (b,c1,c2)
    #use np.delete to 'pop' the nth strategy
    rex = np.reshape(x,(param['n_cos'],param['n_years'],2))
    print 'rex',rex

    #print np.delete(x,0,0)
    chex = []
    for i in range(param['n_cos']):
        temp = reaction(np.delete(rex,i,0),param['consts'][i],param['consts'][-1],i,param['n_cos'],param['n_years'], param['ini_assets'])
        temp = temp.reshape(param['n_years'],2)
        chex.append(temp)
    #print 'chex',np.array(chex)

    reso = np.array(chex) - np.array(rex)
    resid = sum(sum(sum(np.absolute(reso))))
    #print 'reso all', reso
    print 'resid', resid
    #print 'text',np.multiply(np.array(rex),np.add(np.ones((2,3,2)),np.divide(reso,np.array(rex))))
    return np.add(np.array(rex),reso*0.1), resid
    #return np.array(chex), resid

def test():
    n_companies = 15
    n_years = 3
    state_0 = np.full((n_companies,n_years,2),0.0)
    #state_0 = np.random.rand(n_companies,n_years,2)*5.0
    consts = [0.0 for j in range(n_companies)]+[1.0]
    ini_assets = [
    [0,100,100,100,35,35,35],
    [0,100,100,100,35,35,35]]
    print 'consts',consts

    param = {'consts': consts, 'n_cos': n_companies, 'n_years':n_years, 'ini_assets': ini_assets}
    #x0 = np.array([[0.3, 0.3,0.3],[0.3,0.3,0.3]])
    #i.e. x_arr - f(x_arr) = 0
    ans, resid = vector_reaction_o(state_0, param)
    #print 'ans', ans
    print 'resid', resid
    print ans.reshape(n_companies,n_years,2)

    for ITER in range(100):
    #while resid>0.1:

        ans, resid = vector_reaction_o(ans, param)
        #print 'ans', ans
        print 'resid', resid
        #print ans.reshape(n_companies,n_years,2)
        prof = [0 for k in range(n_companies)]
        price = []
        sums = []
        for j in range(n_years):
            sumL = np.sum(ans,axis=0)[j][0]
            sumM = np.sum(ans,axis=0)[j][1]
            price.append(P(50,sumL,sumM))
            sums.append([sumL,sumM])
            #print sumL
            #print sumM
            for k in range(n_companies):
                prof[k]+=price[j]*(ans[k][j][0]+ans[k][j][1])-20.0*ans[k][j][0]-50.0*ans[k][j][1]
    
        print 'prices: ', price
        print 'sums: ', sums
        print 'prof:', prof
        print 'ITER: ',ITER

    #print vector_reaction(state_0,param)


def main():
    """
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
    curve_low = [5,10,5,10,5]
    curve_med = [10,15,10,15,10]
    curve_hig = [10,5,10,5,10]
    #=135
    print prod_oil(curve_low, curve_med, curve_hig, 130)
    ini_assets = [
    [0,100,100,100,6,7,8],
    [0,100,100,100,6,7,8],
    [0,100,100,100,6,7,8],
    [0,100,100,100,6,7,8],
    [0,100,100,100,6,7,8]]
    state = np.full((5,3,5),0.2)
    print 'state', state
    print sum(state[1][1])
    print state2assets(state,1,1,ini_assets,100)
    """


    #OKAY LETS GIVE IT A GO
    # but okay lets not do that many yet
    n_companies = 2
    n_years = 3
    state_0 = np.full((n_companies,n_years,2),5)
    consts = [0.0 for j in range(n_companies)]+[1.0]
    ini_assets = [
    [0,100,100,100,35,35,35],
    [0,100,100,100,35,35,35]]
    print 'consts',consts

    param = {'consts': consts, 'n_cos': n_companies, 'n_years':n_years, 'ini_assets': ini_assets}
    #x0 = np.array([[0.3, 0.3,0.3],[0.3,0.3,0.3]])
    #i.e. x_arr - f(x_arr) = 0
    ans = optimize.fsolve(vector_reaction, state_0, args = (param), xtol = 1e-8)
    print 'ans'
    print ans
    print ans.reshape(n_companies,n_years,2)
    print vector_reaction(state_0,param)

if __name__ == '__main__':
    #main()
    test()