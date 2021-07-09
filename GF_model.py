import math
import numpy as np
from scipy.stats import binom

# PS_old calculation
def PS_old(n, M):
    value = math.pow((M-1)/M, n-1)
    return value

# PS_new calculation
def PS_new(alpha, n, M):
    value = math.exp((1-n)/(alpha*M))
    return value

# Binomial Distribution (BD)
def BD_cal(N, lamda):
    n_values = list(range(1,N+1))
    BD = [binom.pmf(n-1, N-1, lamda) for n in n_values]
    return BD

def PF(alpha, M, n, j, P_f):
    p_fail = 1-(1-P_f[j])*PS_new(alpha, n, M)
    BD = BD_cal(n, p_fail)
    n_values = list(range(1, n + 1))
    temp = []
    for i in range(n):
        n_ = n_values[i]
        temp.append(BD[i] * PS_new(alpha, n_, M))
    return sum(temp)

# Success Probability at the Dedicated Resource and Shared Resource
def DR_SR_success_with_ack_old(M, N, K, Q, P_f, lamda, alpha):
    BD = BD_cal(N, lamda)
    success_prob_list=[]
    temp_list = []
    temp_prod = []

    ###################################
    # When there are no shared resources
    if M == 0:
        return 1 - math.prod(P_f)

    ###################################
    # When there is at least one shared resource
    for i in range(N):
        n_values = list(range(1,N+1))
        n = n_values[i]

        # K trials per UE (N)
        k_values = list(range(1, K + 1))
        temp_list.clear()
        for j in range(K):
            k = k_values[j]

            ########################### Happening at Dedicated Resource
            if k <= Q:
                if k == 1:  # First
                    temp_list.append(1-P_f[j])
                else:
                    print("hello")
                    temp_prod.clear()
                    for q in range(k):
                        if q+1 == k:  # if it is the last one success
                            temp_prod.append(1-P_f[q]) # success
                        else:        # otherwise it is failure
                            temp_prod.append(P_f[q])    # fail
                    temp_list.append(np.prod(temp_prod) )#   FFFFFS
            ########################## Happening at Shared Resource
            else:
                if k==1: # First
                    temp_list.append((1-P_f[j])*PS_new(alpha, n, M))
                else:
                    temp_prod.clear()
                    for q in range(k):
                        if q+1 == k:                # if it is the last one success
                            temp_prod.append((1-P_f[q]))#*PS_new(alpha, n, M)) # success
                        else:                       # otherwise it is failure
                            # Here are two case
                            # Failure happening at Dedicated
                            # Failure happening at Shared
                            if q <= Q:
                                temp_prod.append(P_f[q+Q-1])                # failure
                            else:
                                temp_prod.append(PF(alpha, M, n, q+Q-1, P_f))    # failure
                    temp_list.append(np.prod(temp_prod) )#   FFFFFS
        success_prob_list.append(BD[i] * sum(temp_list))
    return np.sum(success_prob_list)

# Success Probability at the Dedicated Resource and Shared Resource
def DR_SR_success_with_ack(M, N, K, Q, P_f, lamda, alpha):

    BD = BD_cal(N, lamda)
    n_values = list(range(1,N+1))
    k_values = list(range(1, K + 1))
    temp_k_trial = []
    temp_prod    = []
    success_prob = 0

    for n_index in range(N):
        n_i = n_values[n_index]

        # initialize the temp_list
        temp_k_trial.clear()

        for k_index in range(K):
            k_i = k_values[k_index]

            #####################################################################################
            #####################################################################################
            # Starting from k_i = 1, and so {S, FS, FFS, FFFS, FFFFS, .....}
            # It is the recurrent function
            # For instance, k_i = 2: {S1, FS2==(1-S1)S2} needs to be calculated
            temp_prod.clear()
            #print("--temp_prod.cleared")
            for r_i in range(k_i):
                #####################################################################################
                #####################################################################################
                # Happening at Dedicated Resource (DR)
                if r_i+1 <= Q:
                    # Success, for instance, in {S1, FS2}, it means {S1} or {S2}
                    if r_i == k_index:
                        last_success_prob = 1-P_f[k_index]
                        #print("----------temp_prod.append (DR-S)")
                        temp_prod.append(last_success_prob)
                        # this happens only when it successes lastly
                        #print("------------------------------temp_append.append  (DR-S)")
                        temp_k_trial.append(np.prod(temp_prod))
                    # Failure, for instance, in {S1, F1S2==(1-S1)S2}, F1 = (1-S1)
                    else:
                        failure_prob = P_f[r_i]
                        #print("----------temp_prod.append (DR-F)")
                        temp_prod.append(failure_prob)

                ######################################################################################
                #####################################################################################
                # Happening at Shared Resource (SR)
                else:
                    # Success, for instance, in {S1, FS2}, it means {S1} or {S2}
                    if r_i == k_index:
                        last_success_prob = (1-P_f[k_index]) * PF(alpha, M, n_i, k_index - 1, P_f)
                        #print("----------temp_prod.append (SR-S)")
                        temp_prod.append(last_success_prob)
                        # this happens only when it successes lastly
                        #print("------------------------------temp_append.append  (SR-S)")
                        temp_k_trial.append(np.prod(temp_prod))
                    # Failure, for instance, in {S1, F1S2==(1-S1)S2}, F1 = (1-S1)
                    else:
                        failure_prob = 1 - last_success_prob
                        #print("----------temp_prod.append (SR-F)")
                        temp_prod.append(failure_prob)
        #print(np.sum(temp_k_trial), ",  list: " ,temp_k_trial)
        success_prob = success_prob + BD[n_index] * np.sum(temp_k_trial)
    return (success_prob)

# Success Probability at the Dedicated Resource and Shared Resource
def DR_SR_success_without_ack(M, N, K, Q, P_f, lamda, alpha):
    BD = BD_cal(N, lamda)
    PS  = []
    n_values = list(range(1,N+1))
    PDR_f = P_f[0:Q]
    PSR_f = P_f[Q:]

    ###################################
    # When there are no shared resources
    if M == 0:
        p_s = (1 - math.prod(P_f))
        return p_s
    ###################################
    # When there is at least one shared resource
    for i in range(N):
        n = n_values[i]
        if K <= Q:
            p_s = BD[i] * (1 - math.prod(PDR_f))
        else:
            #print("n: ",n , ",M: ", M, ",PS_new: ", PS_new(alpha, n, M))
            p_s = BD[i] * (1 - math.prod(PDR_f) * math.prod([1-(1-j)*PS_new(alpha, n, M) for j in PSR_f]))
        PS.append(p_s)
    #print(PS)
    return np.sum(PS)

