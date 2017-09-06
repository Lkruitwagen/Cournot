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
    xn = xn.reshape(1,n_years,5) #add a proper len variable
    x = np.insert(x_allmn,n,xn,0)
    #print '\rx',x,
    assets, meta = state2assets(x,1,1,ini_assets,100)
    div = sum([el['div'][n] for el in meta])
    return -1.0*div

def oil_price(curve,dem):
    sup_low = float(curve[0])
    sup_med = float(curve[0])+float(curve[1])
    sup_hig = sup_med+float(curve[2])
    #print sup_low,sup_med,sup_hig
    try:
        y1 = 20.0*(dem/sup_low)*(1/(1+exp(-10.0*dem)))*(1-(1/(1+exp(-10.0*(dem-sup_low)))))
    except:
        y1=0
    try:
        y2 = (20.0+(50.0-20.0)*((dem-sup_low)/(sup_med-sup_low)))*(1/(1+exp(-10.0*(dem-sup_low))))*(1-(1/(1+exp(-10.0*(dem-sup_med)))))
    except:
        y2 = 0
    try:#print y2
        y3 = (50.0+(80.0-50.0)*((dem-sup_med)/(sup_hig-sup_med)))*((1/(1+exp(-10.0*(dem-sup_med)))))*(1-(1/(1+exp(-1.0*(dem-sup_hig))))) #*(1/(1+exp(-10.0*(dem-sup_med))))*(1-(1/(1+exp(-10.0*(dem-sup_hig)))))
    except:
        y3=0
    try:
        y4 = (80.0+(140.0-80.0)*((dem-sup_hig)/(sup_hig*0.2)))*((1/(1+exp(-1.0*(dem-sup_hig)))))*(1-(1/(1+exp(-1.0*(dem-sup_hig*1.2))))) #*(1/(1+exp(-10.0*(dem-sup_hig))))*(1-(1/(1+exp(-10.0*(dem-(sup_hig*1.2))))))
    except:
        y4 = 0
    try:
        y5 = (140.0)*((1/(1+exp(-1.0*(dem-sup_hig*1.2))))) #*(1/(1+exp(-10.0*(dem-sup_hig))))*(1-(1/(1+exp(-10.0*(dem-(sup_hig*1.2))))))
    except:
        y5 = 0
    #print 'y1',y1
    #print 'y2',y2
    #print 'y3',y3
    #print 'y4',y4
    #print 'y5', y5
    y = y1+y2+y3+y4+y5
    #print y
  
    #print 'price'
    if y<0:
        return 0
    else:
        return y
  

def reaction(x_allmn,cn,b,n,n_cos,n_years, ini_assets):
	#xn's best response to x_all-n
    #~~~!!!! need to call -profit differently - return a vector instead of a single val. 
    #xn's best response to x_all-n
    #xn = optimize.brute(lambda x: -profit(x,np.insert(x_allmn,n,x),cn,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    
    b = (0.0,1.0)
    bnds = tuple([b for j in range(n_years*5)])
    #print 'bnds',bnds

    minimizer_kwargs = {'args': (x_allmn,n,cn,b,ini_assets), 'bounds':bnds}
    x0 = np.full((1,n_years,5),0.1)
    #xn = optimize.brute(objective, ((0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)), args = (x_allmn,n,cn,b, ini_assets), finish = optimize.fmin)
    xn = optimize.basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs) #, niter=200)

    #print 'rxn xn:', xn
    print 'xn.x',xn.x, type(xn.x)

    return xn.x

def vector_reaction(x,param): # vector param = (b,c1,c2)
    #use np.delete to 'pop' the nth strategy
    rex = np.reshape(x,(param['n_cos'],param['n_years'],5))
    print 'rex',rex

    #print np.delete(x,0,0)
    chex = []
    for i in range(param['n_cos']):
        temp = reaction(np.delete(rex,i,0),param['consts'][i],param['consts'][-1],i,param['n_cos'],param['n_years'], param['ini_assets'])
        temp = temp.reshape(param['n_years'],5)
        chex.append(temp)
    print 'chex',np.array(chex)

    reso = np.array(rex) - np.array(chex)
    resid = sum(sum(sum(np.absolute(reso))))
    print 'reso all', reso
    print 'resid', resid
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



    #OKAY LETS GIVE IT A GO
    # but okay lets not do that many yet
    n_companies = 2
    n_years = 3
    state_0 = np.full((n_companies,n_years,5),0.2)
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
    print ans.reshape(n_companies,n_years,5)

if __name__ == '__main__':
    main()