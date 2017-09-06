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
#0217.07.12 - renormalize REX, try a minimizer instead of basinhopping, add floor to cash
#2017.07.13 - negative cash coming from LIFTING COSTS! So much free DevL! What???
#2017.08.29 - stretch out upper price bound. See 10b for smaller iteration step.


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
    #print 'hi'

    

    P = oil_price(sum(curve_low),sum(curve_med),sum(curve_hig),dem)
    #if isnan(P):
    #    print 'caught a Pnan over here'
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

def print_assets(assets,state,reso):
    n_cos = len(assets)
    n_years = len(assets[0])
    n_assets = len(assets[0][0])
    #print reso

    for co in range(n_cos):
        print 'company number {}'.format(co)
        print '{:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} {:>7} |  {:>7} {:>7} {:>7} {:>7} {:>7} '.format('yr','cash','UndL','UndM','UndH','DevL','DevM','DevH','Div', 'Exp', 'UpgL', 'UpgM','UpgH','rDiv', 'rExp', 'rUpgL', 'rUpgM','rUpgH')
        for y in range(n_years):
            print '{:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} {:>7} |  {:>7} {:>7} {:>7} {:>7} {:>7} '.format(y, round(assets[co][y][0],1),
                round(assets[co][y][1],1),
                round(assets[co][y][2],1),
                round(assets[co][y][3],1),
                round(assets[co][y][4],1),
                round(assets[co][y][5],1),
                round(assets[co][y][6],1),
                int(state[co][y][0]*100),
                int(state[co][y][1]*100),
                int(state[co][y][2]*100),
                int(state[co][y][3]*100),
                int(state[co][y][4]*100),
                round(reso[co][y][0]*100,1),
                round(reso[co][y][1]*100,1),
                round(reso[co][y][2]*100,1),
                round(reso[co][y][3]*100,1),
                round(reso[co][y][4]*100,1))


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
            meta.append({'P':P, 'cash': cash_gen, 'div':[0 for i in range(n_cos)],'prod':[[0,0,0],[0,0,0]]})

            for co in range(n_cos):
                # do cash [0]
                assets[co][y][0] = cash_gen[co]

                # do devL[4]
                assets[co][y][4] = ini_assets[co][4]# - prod_low[co]

                # do devM[5]
                assets[co][y][5] = ini_assets[co][5]# - prod_med[co]

                #do devH[6]
                assets[co][y][6] = ini_assets[co][6]# - prod_hig[co]

                #do undL
                assets[co][y][1] = ini_assets[co][1]

                #do undM
                assets[co][y][2] = ini_assets[co][2]

                #do undH
                assets[co][y][3] = ini_assets[co][3]

            #if isnan()
    
        elif y>0:
            #do stuff with y-1
            curve_low = [el[y-1][4] for el in assets]
            #print 'curve_low', [el[y-1][4] for el in assets]
            curve_med = [el[y-1][4] for el in assets]
            curve_high = [el[y-1][4] for el in assets]

            P, prod_low, prod_med, prod_hig, cash_gen = prod_oil(curve_low, curve_med, curve_high, dem)
            #if isnan(P):
            #    print 'here dat Pnan'
            meta.append({'P':P, 'cash': cash_gen, 'div':[assets[co][y-1][0]*state[co][y-1][0] for co in range(n_cos)],'prod':[prod_low,prod_med,prod_hig]})
            
    
        #then do assets in cos:
            for co in range(n_cos):
                #herro
                #error catching: use if statements to limit>=0
                #Add upg delay
                # do cash [0]
                #print 'sum', sum(state[co][y-1])
                #need a negative block on cash probably
                if sum(state[co][y-1])>1:
                    #print '>1, why?'
                    #print state[co][y-1], sum(state[co][y-1])
                    #raw_input('-->')
                    assets[co][y][0] = cash_gen[co]
                else:
                    assets[co][y][0] = assets[co][y-1][0]*(1.0-sum(state[co][y-1])) + cash_gen[co]

                # do devL[4]
                if (assets[co][y-1][4] - prod_low[co] + assets[co][y-1][0]*state[co][y-1][2]/cap_upg_low)<0:
                    assets[co][y][4] = 0
                else: #if the amount being upgraded is greater than last year's undL
                    if (assets[co][y-1][0]*state[co][y-1][2]/cap_upg_low)>assets[co][y-1][1]:
                        assets[co][y][4] = assets[co][y-1][4] - prod_low[co] + assets[co][y-1][1]
                    else:
                        assets[co][y][4] = assets[co][y-1][4] - prod_low[co] + assets[co][y-1][0]*state[co][y-1][2]/cap_upg_low

                # do devM[5]
                if (assets[co][y-1][5] - prod_med[co] + assets[co][y-1][0]*state[co][y-1][3]/cap_upg_med)<0:
                    assets[co][y][5] = 0
                else: #if the amount being upgraded is greater than last years'undM
                    if (assets[co][y-1][0]*state[co][y-1][3]/cap_upg_med)>assets[co][y-1][2]:
                        assets[co][y][5] = assets[co][y-1][5] - prod_med[co] +assets[co][y-1][2]
                    else:
                        assets[co][y][5] = assets[co][y-1][5] - prod_med[co] + assets[co][y-1][0]*state[co][y-1][3]/cap_upg_med

                #do devH[6]
                if (assets[co][y-1][6] - prod_hig[co] + assets[co][y-1][0]*state[co][y-1][4]/cap_upg_hig)<0:
                    assets[co][y][6] =0
                else:#if the amount being upgraded is greater than last year's undH
                    if (assets[co][y-1][0]*state[co][y-1][4]/cap_upg_hig)>assets[co][y-1][3]:
                        assets[co][y][6] = assets[co][y-1][6] - prod_hig[co] + assets[co][y-1][3]
                    else:
                        assets[co][y][6] = assets[co][y-1][6] - prod_hig[co] + assets[co][y-1][0]*state[co][y-1][4]/cap_upg_hig

                if (assets[co][y-1][0]*state[co][y-1][1])<0:
                    exp_bbl = 0
                else:
                    exp_bbl = assets[co][y-1][0]*state[co][y-1][1]/(por_low*cap_exp_low+por_med+cap_exp_med+por_hig*cap_exp_hig)
                
                #do undL
                if (assets[co][y-1][1] - assets[co][y-1][0]*state[co][y-1][2]/cap_upg_low + exp_bbl*por_low)<0:
                    assets[co][y][1] = 0
                else:
                    assets[co][y][1] = assets[co][y-1][1] - assets[co][y-1][0]*state[co][y-1][2]/cap_upg_low + exp_bbl*por_low

                #do undM
                if (assets[co][y-1][2] - assets[co][y-1][0]*state[co][y-1][3]/cap_upg_med + exp_bbl*por_med)<0:
                    assets[co][y][2] = 0
                else:
                    assets[co][y][2] = assets[co][y-1][2] - assets[co][y-1][0]*state[co][y-1][3]/cap_upg_med + exp_bbl*por_med

                #do undH
                if (assets[co][y-1][3] - assets[co][y-1][0]*state[co][y-1][4]/cap_upg_hig + exp_bbl*por_hig)<0:
                    assets[co][y][3] = 0
                else:
                    assets[co][y][3] = assets[co][y-1][3] - assets[co][y-1][0]*state[co][y-1][4]/cap_upg_hig + exp_bbl*por_hig
    #print_assets(assets)

    y = n_years
    curve_low = [el[y-1][4] for el in assets]
            #print 'curve_low', [el[y-1][4] for el in assets]
    curve_med = [el[y-1][4] for el in assets]
    curve_high = [el[y-1][4] for el in assets]

    P, prod_low, prod_med, prod_hig, cash_gen = prod_oil(curve_low, curve_med, curve_high, dem)
    meta.append({'P':P, 'cash': cash_gen, 'div':[assets[co][y-1][0]*state[co][y-1][0] for co in range(n_cos)], 'prod':[prod_low,prod_med,prod_hig]})
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
    #print 'n',n
    n_years = x_allmn.shape[1]
    xn = xn.reshape(1,n_years,5) #add a proper len variable
    x = np.insert(x_allmn,n,xn,0)
    #print 'xn',xn
    assets, meta = state2assets(x,1,1,ini_assets,100.0)
    #print 'assets',assets
    #print meta
    total_div = sum([el['div'][n] for el in meta])
    #print '\rtotal div',total_div
    #if isnan(total_div):
    #    print 'oh no naaaaaaaannnnnn'
    #    print 'x: ', x
    #    print 'assets: ',assets
    #    print 'meta: ', meta
        #exit(0)
    return total_div*-1.0


    """
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
    """

def oil_price(sumL,sumM,sumH,d):
    A = [20,30,30,60]
    k = -0.3
    x0 = [
    sumL/2.0,
    sumL+sumM/2.0,
    sumL+sumM+sumH/2.0,
    (sumL+sumM+sumH)*1.5
    ]
    try:
      y0 = A[0]/(1.0+exp(k*(d-x0[0])))
    except OverflowError:
      y0 = A[0]
    try:
      y1 = A[1]/(1.0+exp(k*(d-x0[1])))
    except OverflowError:
      y1 = A[1]
    try:
      y2 = A[2]/(1.0+exp(k*(d-x0[2])))
    except OverflowError:
      y2 = A[2]
    try:
      y3 = A[3]/(1.0+exp(k*(d-x0[3])))
    except OverflowError:
      y3 = A[3]
    #y3 = A[3]/(1.0+exp(k*(d-x0[3])))
    y = y0+y1+y2+y3  #+y3
    #print y,'\r',
    #if isnan(y):
    #    print 'oh no  a ynan'
    #    print sumL,sumM,sumH
        #exit(0)
    return y
  
def const_1(x,yr,n_years):
    #sum of allocations in each yr <=1
    #print 'x',x
    #x-1 > 0 => x>1
    #x+1 >0 => x>-1
    #-x -1 >0 => x<-1
    #-x+1>0 => x<1
    x = x.reshape(n_years,5)
    #print 'x', x
    #print x[0]
    #print bug
    #print 'n',n
    #print 'ret', x[0][n],x[1][n],x[2][n],30
    #print -x[0][n]-x[1][n]-x[2][n]+30.0
    #return -x[0][n]-x[1][n]-x[2][n]+40.0
    #return x[0][n]+x[1][n]+x[2][n]-40.0
    return -x[yr][0]-x[yr][1]-x[yr][2]-x[yr][3]-x[yr][4]+1.0



def const_2(x,n):
    #lower bound is 0
    return x[n]-0.0

def const_3(x,n):
    #upper bound is 1.0
    return -x[n]+1.0

def const_4(x,n_alloc,yr,n_years):
    x = x.reshape(n_years,5)
    #smooth is all out => +- 5% yoy
    #<5%
    #
    return -1.0*abs(x[yr][n_alloc]-x[yr-1][n_alloc])+0.05#/x[yr-1][n_alloc]+0.05

def reaction(x_allmn,cn,b,n,n_cos,n_years, ini_assets):
	#xn's best response to x_all-n
    #~~~!!!! need to call -profit differently - return a vector instead of a single val. 
    #xn's best response to x_all-n
    #xn = optimize.brute(lambda x: -profit(x,np.insert(x_allmn,n,x),cn,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    
    b = (0.0,1.0)
    bnds = tuple([b for j in range(n_years*5)])
    #print 'bnds',bnds
    cons = ([])
    #for i in range(2):
        #neither low or med can >30
    #    cons.append({'type': 'ineq', 'fun': con_L,'args':(i,)})
    for yr in range(n_years):
        #sum of allocs in each year<=1
        cons.append({'type': 'eq', 'fun': const_1,'args':(yr,n_years)})
    
    for i in range(n_years*5):
        cons.append({'type': 'ineq', 'fun': const_2,'args':(i,)})
        cons.append({'type': 'ineq', 'fun': const_3,'args':(i,)})

    for i in range(1,n_years):
        for j in range(5):
            cons.append({'type': 'ineq', 'fun': const_4,'args':(j,i,n_years)})
    #print cons
    #raw_input('-->')

    minimizer_kwargs = {'args': (x_allmn,n,cn,b,ini_assets), 'constraints':cons, 'bounds': bnds}
    x0 = np.full((1,n_years,5),0.1)
    tpl_bnds = tuple([((0.0,1.0),)*5*n_years])
    print 'tpl_bnds', tpl_bnds[0]
    rranges = (slice(0.0,1.0,0.2))
    #xn = optimize.brute(objective, tpl_bnds[0], Ns=5,args = minimizer_kwargs, finish = optimize.fmin)  #memory errorrr looollll
    xn = optimize.differential_evolution(objective, tpl_bnds[0], args = minimizer_kwargs['args'],disp=True) 
    #basin hopping occasionally nannnning out - why?
    #add nanflag
    #then try a different minimizer
    #xn = optimize.basinhopping(objective, x0, stepsize =0.1, minimizer_kwargs=minimizer_kwargs, disp=True, callback=call) #, niter=200)
    print xn
    raw_input('paused-->')
    #xn = optimize.minimize(objective, x0, args = (x_allmn,n,cn,b,ini_assets), constraints = cons, bounds = bnds) #, niter=200)

    #print 'rxn xn:', xn
    #print 'xn.x',xn.x, type(xn.x)

    return xn.x

def call (x,f,accept):
	print x.reshape(5,5)
	print f

def vector_reaction(x,param): # vector param = (b,c1,c2)
    #use np.delete to 'pop' the nth strategy
    rex = np.reshape(x,(param['n_cos'],param['n_years'],5))
    #print 'rex',rex

    #print np.delete(x,0,0)
    chex = []
    for i in range(param['n_cos']):
        nanflag = True
        nancount = 0
        while nanflag == True:
            #set to true for just initial
            temp = reaction(np.delete(rex,i,0),param['consts'][i],param['consts'][-1],i,param['n_cos'],param['n_years'], param['ini_assets'])
            if sum([isnan(el) for el in temp])>0:
                nancount+=1
                print 'nancount',nancount
                raw_input('nan found  - continue?')
            else:
                nanflag = False


        temp = temp.reshape(param['n_years'],5)
        #print 'returned optim', temp
        #raw_input('continue?')
        chex.append(temp)
    #print 'chex',np.array(chex)

    reso = np.array(chex) - np.array(rex)
    resid = sum(sum(sum(np.absolute(reso))))
    #print 'reso all', reso
    print 'resid', resid
    if resid>10.0:
        output = np.add(np.array(rex),reso*0.2)
    elif resid>2.0:
        output = np.add(np.array(rex),reso*0.05)
    else:
        output = np.add(np.array(rex),reso*0.01)
    #print output
    #output = output/np.sum(output,axis=2)
    #print 'text',np.multiply(np.array(rex),np.add(np.ones((2,3,2)),np.divide(reso,np.array(rex))))

    #additions to rex need to be renormalised
    return output, resid, reso
    #return np.array(chex), resid

def main():
    n_companies = 3
    n_years = 5
    state_ini = np.array([[[.01,.55,.03,0.16,.25]]*n_years]*n_companies)
    print 'state_ini',state_ini
    state_0 = np.full((n_companies,n_years,5),0.1)
    #state_0 = np.random.rand(n_companies,n_years,5)*0.2
    consts = [0.0 for j in range(n_companies)]+[1.0]
    ini_assets = [[0,1,1,0,1,0,0]]*n_companies
    assets, meta =  state2assets(state_ini,1,1,ini_assets,100)
    print 'prices: ', [el['P'] for el in meta]
    print 'sum_div: ', sum(np.array([el['div'] for el in meta]))
    for el in meta:
        print [round(sum(each)) for each in el['prod']]
    print_assets(assets,state_ini,state_ini)
    raw_input('GO!')

    print 'consts',consts

    param = {'consts': consts, 'n_cos': n_companies, 'n_years':n_years, 'ini_assets': ini_assets}
    #x0 = np.array([[0.3, 0.3,0.3],[0.3,0.3,0.3]])
    #i.e. x_arr - f(x_arr) = 0
    ans, resid, reso = vector_reaction(state_0, param)
    #print 'ans', ans
    print 'resid', resid
    print ans.reshape(n_companies,n_years,5)
    

    for ITER in range(160):
    #while resid>0.1:

        ans, resid, reso = vector_reaction(ans, param)
        #print 'ans', ans
        print 'resid', resid
        state = ans.reshape(n_companies,n_years,5)
        assets, meta = state2assets(state,1,1,ini_assets,100.0)
        print_assets(assets,state, reso)
        print 'prices: ', [el['P'] for el in meta]
        print 'sum_div: ', sum(np.array([el['div'] for el in meta]))
        #for el in meta:
        #    print [round(sum(each)) for each in el['prod']]
        #prof = [0 for k in range(n_companies)]
        #price = []
        #sums = []
        #for j in range(n_years):
        #    sumL = np.sum(ans,axis=0)[j][0]
        #    sumM = np.sum(ans,axis=0)[j][1]
        #    price.append(P(50,sumL,sumM))
        #    sums.append([sumL,sumM])
        #    #print sumL
        #    #print sumM
        #    for k in range(n_companies):
        #        prof[k]+=price[j]*(ans[k][j][0]+ans[k][j][1])-20.0*ans[k][j][0]-50.0*ans[k][j][1]
    #
        #print 'prices: ', price
        #print 'sums: ', sums
        #print 'prof:', prof
        #print 'ITER: ',ITER

    #print vector_reaction(state_0,param)
    



if __name__ == '__main__':
    print oil_price(20,20,20,50)
    print oil_price(20,20,20,80)
    print oil_price(20,20,20,72)
    main()