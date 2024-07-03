def is_convertible_to_int(var):
    try:
        int(var)  # Attempt to convert var to int
        return True  # Conversion successful, var can be converted to an integer
    except ValueError:
        return False

###################################################################################################################################################################################################
def generate_lists_v2(input_str, lt,n, p =0,length = None):

    # Split the input into parts
    parts = input_str.split('),')
    parts = [part + ')' if ')' not in part else part for part in parts]  # Ensure closing parenthesis for last part
    # Define the mappings for each character
    part = parts[p]
    length = len(parts)
    char, range_st = part[0], part[2:-1]
    range_str,range_end = range_st.split(',')  # Extract character and range string
  
    if is_convertible_to_int(range_str) and is_convertible_to_int(range_end):
#         print("input at block 1 is:",lt)
        l = int(range_end) - int(range_str) +1
#         print("l is:",l)
        for i in range(l):
            dm_lt = [char]*(i+1)
            lt.append(dm_lt)
        p += 1
#         print("output of first block is:",lt)
        lt = generate_lists_v2(input_str,lt,n, p,length)
        return lt
    
    elif range_str == "0" and range_end== 'n':
        dm_lt = [char]*(n+1)
        lt.append(dm_lt)
#         print("input at n is:",lt)
        return lt
     
    elif range_str == "0" and range_end== 'n-1':
        dm_lt = [char]*n
        lt.append(dm_lt)
        p +=1 
#         print("input at n-1 is:",lt)     
        lt = generate_lists_v2(input_str,lt,n, p,length)
        return lt
    elif range_str == "1" and range_end== 'n-1':
        dm_lt = [char]*(n-1)
        lt[0].extend(dm_lt)
        p +=1 
#         print("input at n-1 is:",lt)     
        lt = generate_lists_v2(input_str,lt,n, p,length)
        return lt
    elif range_str == "1" and range_end== 'n':
        dm_lt = [char]*n
        lt[0].extend(dm_lt)
        p +=1 
        return lt
    
    elif not is_convertible_to_int(range_str) and range_end == "n":
#         print("input in n is:",lt)
        for i in range(len(lt)):
            l = n - len(lt[i]) +1
            dm_lt = [char]*l
            lt[i].extend(dm_lt)
#         print("final output in n is:",lt)
        return lt
        
    elif not is_convertible_to_int(range_str)  and range_end =='n-1':
        for i in range(len(lt)):
            l = (n-1) - len(lt[i]) +1
            dm_lt = [char]*(l)
            lt[i].extend(dm_lt)
        p += 1
        generate_lists_v2(input_str,lt,n, p,length)
        return lt
    
    elif range_str == "0" and is_convertible_to_int(range_end) == 0 :
        
        max_end = (n+1)-length+p+1
        full_lt = []

        for end in range(max_end):
            st_lt = []
            l = end +1
            dm_lt = [char]*l
            st_lt.extend(dm_lt)
            full_lt.append(st_lt.copy())
        lt = full_lt.copy()
        p += 1
        if p < len(parts):
            lt = generate_lists_v2(input_str,lt,n, p,length)
        return lt
    
    else:

        max_end = (n+1)-length+p+1

        full_lt = []
        for i in range(len(lt)):
            st = len(lt[i])

            for end in range(st,max_end):
                st_lt = lt[i].copy()
                l = end - st+1
                dm_lt = [char]*l
                st_lt.extend(dm_lt)
                full_lt.append(st_lt.copy())
        lt = full_lt.copy()
        p += 1
        if p < len(parts):
            lt = generate_lists_v2(input_str,lt,n, p,length)
        return lt

    

###################################################################################################################################################################################################


def generate_numbers(input_lt,n,i=0,out_lt= []):
    
    if input_lt[i]== 'N':
        if out_lt != []:
            inner_list_count = sum(isinstance(item, list) for item in out_lt)
        else:
            inner_list_count = 0
        if inner_list_count ==1:
            out_lt[0].append(0)
        elif inner_list_count ==0:
            out_lt.append([0])
#             print(out_lt)
        elif inner_list_count >1:
            for k in range(inner_list_count):
                out_lt[k].append(0)
        if i < len(input_lt)-1:
            i +=1
            out_lt = generate_numbers(input_lt,n,i,out_lt)
        return out_lt
    if input_lt[i]== 'I':
        if out_lt != []:
            inner_list_count = sum(isinstance(item, list) for item in out_lt)
        else:
            inner_list_count = 0
        if inner_list_count ==1:
            out_lt[0].append(i)
#             print(out_lt)
        elif inner_list_count ==0:
            out_lt.append([i])
#             print(out_lt)
        elif inner_list_count >1:
            for k in range(inner_list_count):
                out_lt[k].append(i)
#             print(out_lt)
        if i < len(input_lt)-1:
            i +=1
            out_lt = generate_numbers(input_lt,n,i,out_lt)
        return out_lt
    if input_lt[i]== 'A':
        if out_lt != []:
            inner_list_count = sum(isinstance(item, list) for item in out_lt)
        else:
            inner_list_count = 0
        if inner_list_count ==1:
            out_lt[0].append(n)
        elif inner_list_count ==0:
            out_lt.append([n])
        elif inner_list_count >1:
            for k in range(inner_list_count):
                out_lt[k].append(n)
        if i < len(input_lt)-1:
            i +=1
            out_lt = generate_numbers(input_lt,n,i,out_lt)
        return out_lt
    if input_lt[i]== 'S':
        possible_values = [z for z in range(1,n)]
        acceptable_values = [item for item in possible_values if item != i]
        if out_lt != []:
            inner_list_count = sum(isinstance(item, list) for item in out_lt)
        else:
            inner_list_count = 0
        if inner_list_count ==1:
            if isinstance(out_lt[0],list):
                out_lt= [item[:] for item in out_lt for _ in range(len(acceptable_values))]
            else:
                out_lt = [out_lt[:] for _ in range(len(acceptable_values))]

            for z in range(len(out_lt)):
                out_lt[z].append(acceptable_values[z])
#                 print(out_lt)
        elif inner_list_count == 0:
            for p in acceptable_values:
                out_lt.append([p])
#             print(out_lt)
        elif inner_list_count >1:
            dm_lt = []
            dm_in_lt = []
            for k in range(inner_list_count):
                lt = out_lt[k].copy()
#                 dm_in_lt = [item[:] for item in lt for _ in range(len(acceptable_values))]
                if isinstance(lt[0],list):
                    dm_in_lt= [item[:] for item in lt for _ in range(len(acceptable_values))]
                else:
                    dm_in_lt = [lt[:] for _ in range(len(acceptable_values))]
#                 print(dm_in_lt)
                for p in range(len(acceptable_values)):
                    dm_in_lt[p].append(acceptable_values[p])
                dm_lt.extend(dm_in_lt)
            out_lt = dm_lt 
#             print(out_lt)
        if i < len(input_lt)-1:
            i +=1
            out_lt = generate_numbers(input_lt,n,i,out_lt)
        
        return out_lt

###################################################################################################################################################################################################


def gen_number_list_of_lists(input_lt,n,out_lt = []):
    dm_in_lt = []
    lt = []
    for i in range(len(input_lt)):
#         print(input_lt[i])
        if len(input_lt[i]) ==1:
            j=0
            lt = generate_numbers(input_lt[i][0],n,j,out_lt= [])
#             print(lt)
            out_lt.append(lt)
            lt = []
        elif len(input_lt[i]) >1:
            for k in range(len(input_lt[i])):
#                 print(input_lt[i][k])
                j=0
                dm_in_lt = generate_numbers(input_lt[i][k],n,j,out_lt= [])
                lt.extend(dm_in_lt)
            out_lt.append(lt)
            lt = []
    return out_lt
###################################################################################################################################################################################################


def create_prm_lt(wrong_pred_df,new_df):
    index_lt = wrong_pred_df.index.tolist()
    df = new_df.loc[index_lt]
    df_iter = df[['theta','beta','c','s','alpha','rho','pi_l','pi','pi_h','gamma']]
    prm_list = [list(row) for row in df_iter.values]
    return prm_list

###################################################################################################################################################################################################
def fac(n):
    if n == 0:
        return 1
    res = 1
    for i in range(1,n+1,1):
        res = res*i
    return res
def com(m, n):
    return float(fac(n))/(fac(n-m)*fac(m))


###################################################################################################################################################################################################

def reverify_pred(prm_list,num_lt,n_pr,M=1000,delta = 0.001,params = {}):
    import pandas as pd
    import time
    import gurobipy as gp
    from gurobipy import GRB
    st = time.process_time()
    env = gp.Env(params=params)
    results_list = []
    p = 0
    count = 0
    for prm in prm_list :
        lt = []
        n_lt = num_lt[p]
        for n in n_lt:
            count +=1
            print("iteration num is:",p,end= "\r")
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

            for i in range(n_pr+1):
                m2.addConstr(q[i] == n[i])
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
            m2.setParam('DualReductions', 1)
            m2.optimize()
            temp = m2.getObjective()
    #         print("model status is:",m2.status)
            if m2.status == 2:
                res_tbr = []; E_r_tbr = 0; E_obj_tbr = 0; E_trees_tbr = 0
                for i in range(n_pr+1):
                    res_tbr.append([round(i,0), round(q[i].X,0), round(r_q[i].X,0), round(temp.getValue(),0)])
                    E_obj_tbr += com(i, n_pr)*pow(pi, i)*pow(1-pi, n_pr-i)*temp.getValue()
                    E_r_tbr += com(i, n_pr)*pow(pi, i)*pow(1-pi, n_pr-i)*r_q[i].X

                    temp_tr = (mu_q[i].X*((1-pi_h)*(n_pr-i)+rho*q[i].X+rho*(n_pr-i))
                               + (1-mu_q[i].X)*((1-pi_l)*(n_pr-q[i].X) + q[i].X - (1-rho)*i+rho*pi_l*(n_pr-q[i].X)) )
                    E_trees_tbr += temp_tr

                res_tbr = pd.DataFrame(res_tbr, columns = ["Infestation", "Treatment", "Reimbursement", "Obj"])
                lt.append(res_tbr)
            else:
                n = [n_pr]*(n_pr+1)
                m2.optimize()
                if m2.status == 2:
                    res_tbr = []; E_r_tbr = 0; E_obj_tbr = 0; E_trees_tbr = 0
                    for i in range(n_pr+1):
                        res_tbr.append([round(i,0), round(q[i].X,0), round(r_q[i].X,0), round(temp.getValue(),0)])
                        E_obj_tbr += com(i, n_pr)*pow(pi, i)*pow(1-pi, n_pr-i)*temp.getValue()
                        E_r_tbr += com(i, n_pr)*pow(pi, i)*pow(1-pi, n_pr-i)*r_q[i].X

                        temp_tr = (mu_q[i].X*((1-pi_h)*(n_pr-i)+rho*q[i].X+rho*(n_pr-i))
                                + (1-mu_q[i].X)*((1-pi_l)*(n_pr-q[i].X) + q[i].X - (1-rho)*i+rho*pi_l*(n_pr-q[i].X)) )
                        E_trees_tbr += temp_tr
                    res_tbr = pd.DataFrame(res_tbr, columns = ["Infestation", "Treatment", "Reimbursement", "Obj"])
                    lt.append(res_tbr)
                elif m2.status == 4:
                    m2.setParam('DualReductions', 0)
                    m2.optimize()
                    updated_status = m2.status
                    if updated_status == 3:
                        status = "infeasible"
                        lt.append(status)  
                    else:
                        status = "unbounded/infeasible"
                        lt.append(status)
                elif m2.status == 3:

                    status = "infeasible"
                    lt.append(status)


        results_list.append(lt)
        p += 1  

    end = time.process_time()

    resolve_time = end- st
    
    return results_list,resolve_time, count

###################################################################################################################################################################################################

def create_op_val_lt(results_list):
    import pandas as pd
    final_lt = []

    for i in range(len(results_list)):
        lt = []
        for j in results_list[i]:
            if isinstance(j,str):
                lt.append(j)
            if isinstance(j,pd.DataFrame):
                value = j.at[0, 'Obj']
                lt.append(value)
        final_lt.append(lt)
    return final_lt


###################################################################################################################################################################################################

def create_pred_opt_lt(final_lt):
    import pandas as pd
    pred_opt_lt = []

    # Function to process the list of lists according to the given rules
    def is_convertible_to_int(var):
        try:
            int(var)  # Attempt to convert var to int
            return True  # Conversion successful
        except ValueError:
            return False  # Conversion failed
###################################################################################################################################################################################################
    def process_list_of_lists(input_list):
        output_list = []
        for inner_list in input_list:
            if not inner_list:
                output_list.append(None)
                continue

            # Process each element, checking if it can be converted to an integer
            integers = [int(item) for item in inner_list if is_convertible_to_int(item)]
            strings = [item for item in inner_list if not is_convertible_to_int(item)]

            if integers and not strings:
                output_list.append(max(integers))
            elif strings and not integers:
                output_list.append(strings[0])
            elif integers:
                output_list.append(max(integers))
            else:
                output_list.append(None)

        return output_list


    # Process the example input list
    pred_opt_lt = process_list_of_lists(final_lt)

    return pred_opt_lt
###################################################################################################################################################################################################

def create_wrong_pred_df(final_lt_all,wrong_pred_df,new_df,pred_opt_lt):
    
    import pandas as pd
    opt_gap_lt = []
    final_lt = []
    ind_lt = wrong_pred_df.index.tolist()
    
    for i in ind_lt:
        final_lt.append(final_lt_all[i])
        
    for i in range(len(pred_opt_lt)):
        if isinstance(pred_opt_lt[i],int):
            l = abs(final_lt[i]-pred_opt_lt[i])/abs(final_lt[i]) if int(final_lt[i]) != 0 else None
            opt_gap_lt.append(l)
        else:
            opt_gap_lt.append(pred_opt_lt[i])

    wrong_pred_df['opt_gap'] = opt_gap_lt
    wrong_pred_df['partition'] = wrong_pred_df.apply(lambda row:
                                       'base zero' if pd.isnull(row['opt_gap']) else
                                       'infeasible' if row['opt_gap'] == 'infeasible' else
                                       '0' if row['opt_gap']*100 == 0 else
                                       '(0,10]' if 0 < row['opt_gap']*100 <= 10 else
                                       '(10,25]' if 10 < row['opt_gap']*100 <= 25 else
                                       '(25,50]' if 25 < row['opt_gap']*100 <= 50 else
                                       '(50,75]' if 50 < row['opt_gap']*100 <= 75 else
                                       '(75,100]' if 75 < row['opt_gap']*100 <= 100 else
                                       '> 100' if row['opt_gap']*100 > 100 else
                                       'unknown', axis=1)
    
    opt_df = wrong_pred_df[wrong_pred_df['partition'] != 'infeasible']

    gap_lt = opt_df['opt_gap'].tolist()

    ind_lt = opt_df.index.tolist()
    gap = 0
    for i in range(len(gap_lt)):
        if gap_lt[i]:
            gap += gap_lt[i]
        if gap_lt[i] is None:
            gap += 1.5 
    final = gap/new_df.shape[0]
    return wrong_pred_df,final
###################################################################################################################################################################################################

def results_eval(test_df,original_solution, predicted_solution,n,original_parm_df,params= {}):
    test_df['pred_check'] = test_df['original_solution'] == test_df['predicted_solution']
    wrong_pred_df = test_df[ test_df['pred_check']== False]
    lt = []
    pred_lt = wrong_pred_df.sati_pred.tolist()
    for cl in pred_lt:
        lt1= []
        result = generate_lists_v2(cl,lt1 ,n, p =0,length = None)
        lt.append(result)
    
    num_lt = gen_number_list_of_lists(lt,n,out_lt = [])
    prm_list = create_prm_lt(wrong_pred_df,original_parm_df)
    results_list,resolve_time, count = reverify_pred(prm_list,num_lt,n,params)
    final_lt = create_op_val_lt(results_list)
    pred_opt_lt = create_pred_opt_lt(final_lt)
    final_lt_all = original_parm_df.Obj.tolist()
    wrong_pred_df,final = create_wrong_pred_df(final_lt_all,wrong_pred_df,original_parm_df,pred_opt_lt)

    return wrong_pred_df,final,resolve_time
###################################################################################################################################################################################################
