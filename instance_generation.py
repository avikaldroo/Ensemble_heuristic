import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd

def fac(n):
    if n == 0:
        return 1
    res = 1
    for i in range(1,n+1,1):
        res = res*i
    return res
def com(m, n):
    return float(fac(n))/(fac(n-m)*fac(m))

def generate_inputs():
    beta_list = np.random.uniform(150,500,100)
    theta_list = np.random.uniform(50,250,100)
    s_list = np.random.uniform(80,125,100)
    alpha_list = np.random.uniform(30,60,100)
    rho_list = np.random.uniform(0.2,0.8,100)
    gamma_list = np.random.uniform(50,150,100)
    i_pi_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    i_pi_l_list = []
    i_pi_h_list = []
    for p in i_pi_list:
        d_lt = []
        d =0
        while d <=p:
            d_lt.append(round(d,1))
            d += 0.1
        i_pi_l_list.append(d_lt)
    for p in i_pi_list:
        d_lt = []
        d = p
        while d <= 1:
            d_lt.append(round(d,1))
            d += 0.1
        i_pi_h_list.append(d_lt)

    f_list = [3,3.5,4,4.5,5,5.5,6]
    sum = 0
    for k in range(len(i_pi_list)):
        pr = len(i_pi_l_list[k])* len(i_pi_h_list[k])
        sum += pr
    pi_list = [0]*sum
    pi_l_list = [0]*sum
    pi_h_list = [0]*sum
    ix = 0
    for h in range(len(i_pi_list)):
        for i in range(len(i_pi_l_list[h])):
            for j in range(len(i_pi_h_list[h])):
                pi_list[ix] = i_pi_list[h]
                pi_l_list[ix] = i_pi_l_list[h][i]
                pi_h_list[ix] = i_pi_h_list[h][j]
                ix += 1

    prm_list = []
    dummy_list = [0]*10

    for f in f_list:
        k = 0
        c_list = f* beta_list
        for x,y,z in zip(pi_l_list,pi_list,pi_h_list):
            for k in range(len(beta_list)):
                dummy_list = [theta_list[k],beta_list[k],c_list[k],s_list[k],alpha_list[k],rho_list[k],x,y,z,gamma_list[k]]
                prm_list.append(dummy_list)
        
    return prm_list

def tbr_model(prm_list,env = {}):
    results_list = []
    p = 0
    for prm in prm_list:
        theta = prm[0]
        beta  = prm[1]
        c     = prm[2]
        s     = prm[3]
        alpha = prm[4]
        rho   = prm[5]
        pi_l  = prm[6]
        pi    = prm[7]
        pi_h  = prm[8]
        gamma = prm[9]
        # Create a new model
        m2 = gp.Model("quadratic",env=env)
        m2.Params.LogToConsole = 0
        theta_lt = [theta]*(n_pr+1)
        beta_lt  =  [beta]*(n_pr+1)
        c_lt     = [c]*(n_pr+1)
        s_lt     = [s]*(n_pr+1)
        alpha_lt = [alpha]*(n_pr+1)
        rho_lt   = [rho]*(n_pr+1)
        pi_l_lt  = [pi_l]*(n_pr+1)
        pi_lt    = [pi]*(n_pr+1)
        pi_h_lt  = [pi_h]*(n_pr+1)
        gamma_lt = [gamma]*(n_pr+1)
        # Create variables
        mu = m2.addVars(n_pr+1, n_pr+1, vtype=GRB.BINARY,name="mu")
        mu_c = m2.addVars(n_pr+1, n_pr+1, vtype=GRB.BINARY,name="mu_c")
        mu_q = m2.addVars(n_pr+1, vtype=GRB.BINARY,name="mu_q")
        q = m2.addVars(n_pr+1,lb=0, ub=n_pr, vtype=GRB.INTEGER, name = "q")
        r = m2.addVars(n_pr+1, lb=0, ub=float('inf'), name = "r")
        r_q = m2.addVars(n_pr+1, lb=0, ub=float('inf'), name = "r_q")
        phi = m2.addVars(n_pr+1,n_pr+1 ,lb=float('-inf'), ub=float('inf'), name = "phi")
        phi_q = m2.addVars(n_pr+1, lb=float('-inf'), ub=float('inf'),name = "phi_q")
        #utility of the forest service
        u = m2.addVars(n_pr+1,lb=float('-inf'), ub=float('inf'), name = "u") 

        mu_q = m2.addVars(n_pr+1, vtype=GRB.BINARY,name="mu_q")

        m2.setObjective(gp.quicksum(com(i,n_pr)*pow(pi,i)*pow(1-pi,n_pr-i)*u[i] for i in range(n_pr+1)), GRB.MAXIMIZE) 
        #government utility
        for i in range(n_pr+1):
            m2.addConstr(u[i] == -r_q[i] + mu_q[i]*(s*(rho*q[i]+(1-pi_h)*(n_pr-i)+rho*pi_h*(n_pr-i)) - gamma*(i-rho*q[i]+(1-rho)*pi_h*(n_pr-i)))
                    +(1-mu_q[i])*(s*(q[i]-(1-rho)*i+(1-pi_l)*(n_pr-q[i])+rho*pi_l*(n_pr-q[i])) - gamma*((1-rho)*i+(1-rho)*pi_l*(n_pr-q[i])))) 

        #q, r, and phi
        for i in range(n_pr+1):
            m2.addConstr(gp.quicksum(mu_c[i,j] for j in range(n_pr+1)) == 1)
            m2.addConstr(q[i] == gp.quicksum(mu_c[i,j]*j for j in range(n_pr+1)) )
            m2.addConstr(r_q[i] == gp.quicksum(mu_c[i,j]*r[j] for j in range(n_pr+1)) )
            m2.addConstr(phi_q[i] == gp.quicksum(mu_c[i,j]*phi[i,j]  for j in range(n_pr+1)) )

        #IR
        #binary decision variables
        for i in range(n_pr+1):
            m2.addConstr(q[i] >= i - M*mu_q[i])
            m2.addConstr(q[i] <= i + M*(1-mu_q[i]) + delta)
            for j in range(n_pr+1):
                if j >= i:
                    m2.addConstr(mu[i,j] == 0)
                else:
                    m2.addConstr(mu[i,j] == 1)

        phi_0 = 0
        for i in range(n_pr+1):
            if i ==0:
                phi_0 += com(i,n_pr)*pow(pi,i)*pow(1-pi,n_pr-i)*(theta*(1-pi_l)*n_pr-c*pi_l*n_pr)
            elif i == n_pr:
                phi_0 += com(i,n_pr)*pow(pi,i)*pow(1-pi,n_pr-i)*(-c*n_pr)
            else:
                phi_0 += com(i,n_pr)*pow(pi,i)*pow(1-pi,n_pr-i)*(theta*(1-pi_h)*(n_pr-i) - c*(i+pi_h*(n_pr-i)))

            for j in range(n_pr+1):
                w1 = rho*j+(1-pi_h)*(n_pr-i)+rho*pi_h*(n_pr-i)
                t1 = j+pi_h*(n_pr-i)
                k1 = i-rho*j+(1-rho)*pi_h*(n_pr-i)

                w0 = j-(1-rho)*i+(1-pi_l)*(n_pr-j)+rho*pi_l*(n_pr-j)
                t0 = j+pi_l*(n_pr-j)
                k0 = (1-rho)*i + (1-rho)*pi_l*(n_pr-j)

                m2.addConstr(phi[i,j] == mu[i,j]*(theta*w1 - alpha*n_pr - beta*t1 -c*k1)
                            + (1-mu[i,j])*(theta*w0 - alpha*n_pr - beta*t0 -c*k0) + r[j])

        m2.addConstr(gp.quicksum(com(i,n_pr)*pow(pi,i)*pow(1-pi,n_pr-i)*phi_q[i] for i in range(n_pr+1)) >= phi_0)


        #IC
        for i in range(n_pr+1):
            for j in range(n_pr+1):
                m2.addConstr(phi_q[i] >= phi[i,j])

                if i != n_pr:
                    m2.addConstr(r[i+1] >= r[i])

        m2.optimize()
        temp = m2.getObjective()

        res_tbr = []; E_r_tbr = 0; E_obj_tbr = 0; E_trees_tbr = 0
        for i in range(n_pr+1):
            res_tbr.append([round(i,0), round(q[i].X,0), round(r_q[i].X,0), round(temp.getValue(),0)])
            E_obj_tbr += com(i, n_pr)*pow(pi, i)*pow(1-pi, n_pr-i)*temp.getValue()
            E_r_tbr += com(i, n_pr)*pow(pi, i)*pow(1-pi, n_pr-i)*r_q[i].X

            temp_tr = (mu_q[i].X*((1-pi_h)*(n_pr-i)+rho*q[i].X+rho*(n_pr-i))
                    + (1-mu_q[i].X)*((1-pi_l)*(n_pr-q[i].X) + q[i].X - (1-rho)*i+rho*pi_l*(n_pr-q[i].X)) )
            E_trees_tbr += temp_tr



        res_tbr = pd.DataFrame(res_tbr, columns = ["Infestation", "Treatment", "Reimbursement", "Obj"])


        res_tbr['theta'] = theta_lt
        res_tbr['beta']  = beta_lt
        res_tbr['c']     = c_lt
        res_tbr['s']     = s_lt
        res_tbr['alpha'] = alpha_lt
        res_tbr['rho']   = rho_lt
        res_tbr['pi_l']  = pi_l_lt
        res_tbr['pi']    = pi_lt
        res_tbr['pi_h']  = pi_h_lt
        res_tbr['gamma'] = gamma_lt
        results_list.append(res_tbr)
        p += 1

    final_df = pd.concat(results_list, ignore_index=True)
    final_df['Action'] = final_df.apply(lambda row: 'A' if row['Treatment'] == n_pr else ('I' if (row['Treatment'] == row['Infestation']) and (row['Treatment'] > 0)  else ('N' if row['Treatment'] == 0 else  ('S' if row['Treatment'] < row['Infestation'] else 'P')), axis=1)
    return final_df
########################################################################################################################################################################################################################################################################################################
def convert_column_to_lists(df, column_name):
    column_data = df[column_name].tolist()
    result = []

    for i in range(0, len(column_data), 6):
        sublist = column_data[i:i+6]
        result.append(sublist)

    return result
########################################################################################################################################################################################################################################################################################################
def create_separate_list(input_list):
    result_list = []
    
    for inner_list in input_list:
        inner_result = []
        start_index = 0
        prev_elem = inner_list[0]
        
        for i in range(1, len(inner_list)):
            if inner_list[i] != prev_elem:
                if start_index == i - 1:
                    inner_result.append(f"{prev_elem}({start_index})")
                else:
                    inner_result.append(f"{prev_elem}({start_index},{i-1})")
                start_index = i
                prev_elem = inner_list[i]
        
        # Handle the last element in the inner list
        if start_index == len(inner_list) - 1:
            inner_result.append(f"{inner_list[-1]}({start_index})")
        else:
            inner_result.append(f"{inner_list[-1]}({start_index},{len(inner_list)-1})")
        
        # Check if the inner list contains only one unique element
        if len(set(inner_list)) == 1:
            element = inner_list[0]
            indices = f"{element}({start_index},{start_index})"
            inner_result = [indices] * len(inner_list)
        
        result_list.append(inner_result)
    
     return result_list
####################################################################################################################################################
def check_all_same(lst):
    if len(lst) <= 1:
        return True
    first_element = lst[0]
    return all(element == first_element for element in lst[1:])
####################################################################################################################################################

def process_list(input_list):
        
    result = []
    last_index = len(input_list) - 1

    for i in range(len(input_list)):
        a = ''
        l = len(input_list[i])
        if check_all_same(input_list[i]) == 1:
            a = input_list[i][0][0]+'('+'0'+','+'n'+')'
        if check_all_same(input_list[i]) == 0:
            if l ==2:
                l9 = len(input_list[i][0])
                l10 = len(input_list[i][1])
                if (l9 ==6 and l10 ==6):
                    a += input_list[i][0][0]+'('+'0'+','+'j'+')'+','+ input_list[i][1][0]+'('+'j+1'+','+'n'+')'
                if (l9 ==4 and l10 ==6):
                    a += input_list[i][0][0]+'('+'0'+','+'0'+')'+','+ input_list[i][1][0]+'('+'1'+','+'n'+')' 
                if (l9 ==6 and l10 ==4):
                    a += input_list[i][0][0]+'('+'0'+','+'n-1'+')'+','+ input_list[i][1][0]+'('+'n'+','+'n'+')'
            if l==3:
                    l6 = len(input_list[i][0])
                    l7 = len(input_list[i][1])
                    l8 = len(input_list[i][2])
                    if (l6 == 4):
                        a += input_list[i][0][0]+'('+'0'+','+'0'+')'+','
                    if (l6 == 6):
                        a += input_list[i][0][0]+'('+'0'+','+'j'+')'+','
                    if (l6 ==4 and l8 == 6):
                        a += input_list[i][1][0]+'('+'1'+','+'j'+')'+','+input_list[i][2][0]+'('+'j+1'+','+'n'+')'  
                    if (l6 ==4 and l7 == 6 and l8 == 4):
                        a += input_list[i][1][0]+'('+'1'+','+'n-1'+')'+','+input_list[i][2][0]+'('+'n'+','+'n'+')'   
                    if (l6 ==6 and l8 == 6):
                        a += input_list[i][1][0]+'('+'j+1'+','+'k'+')'+','+ input_list[i][2][0]+'('+'k+1'+','+'n'+')' 
                    


            if l >3:
                z = 0
                for k in range(l):
                    l1 = len(input_list[i][k])
                    ind = ['j','k','l','m','p','q','r']
                    if k == 0 :
                        if l1 ==6:
                            a += input_list[i][k][0]+'('+'0'+','+ ind[z]+')'+','
                            z +=1
                        if l1 ==4:
                            a += input_list[i][k][0]+'('+'0'+','+'0'+')'+','
                    if k==1:
                        l10 = len(input_list[i][0])
                        if l10 ==6:
                            a += input_list[i][k][0]+'('+ind[z-1]+'+1'+','+ ind[z]+')'+','
                            z+= 1
                        if l10 ==4:
                            a += input_list[i][k][0]+'('+'1'+','+ ind[z]+')'+','
                            z +=1
                    if ( k > 1 and k < l-2):
                        a += input_list[i][k][0]+'('+ind[z-1]+'+1'+','+ ind[z]+')' +','
                        z+=1

                    if (  k == l-2): #l1 == 6 and
                        l2 = len(input_list[i][k+1])
                        if l2 ==6:
                            a += input_list[i][k][0]+'('+ind[z-1]+'+1'+','+ind[z]+')' +','
                            z += 1
                        if l2== 4:
                            a += input_list[i][k][0]+'('+ind[z-1]+'+1'+','+'n-1'+')'+','
                            z +=1
#                     if ( l1 == 4 and k == l-2):
#                         a += input_list[i][k][0]+'('+'n-1'+','+'n-1'+')'


                    if ( l1 == 6 and k == l-1):
                        l5 = len(input_list[i][k-1])
                        if l5 == 6:
                            a += input_list[i][k][0]+'('+ind[z-1]+'+1'+','+'n'+')'
                    if ( k > 0 and l1 == 4 and k == l-1):
                        a += input_list[i][k][0]+'('+'n'+','+'n'+')'
        result.append(a)
    return result         
####################################################################################################################################################
        
def generate_df(final_df,n_pr):
    p = convert_column_to_lists(final_df, 'Action')
    y = create_separate_list(p)
    z = process_list(y)

    l = [0]* len(final_df['Action'].tolist())

    for u in range(len(final_df['Action'].tolist())):
        k = mt.floor(u / (n_pr+1))
        l[u] = z[k] 
    final_df['Cluster'] = l
    return final_df







