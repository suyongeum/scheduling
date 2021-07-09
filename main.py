# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
from scipy.stats import binom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import GF_model
import multiprocessing

def plot_success_prob_BLER_k(N,Q, M, lamda, alpha, P_f):
    # Get some results
    num_X = 6
    prob_success, K = (num_X, 5)
    arr_result = [[0 for i in range(prob_success)] for j in range(K)]

    result = []
    X_points = []
    for i in range(K):
        result.clear()
        P_f.clear()
        k_now = i + 1
        for j in range(num_X):
            P_f_1 = 1e-1 ** j
            if i==0:
                X_points.append(P_f_1)
            P_f = [P_f_1 * math.pow(1e-01, i) for i in range(K)]  # initial faiures
            arr_result[i][j] = GF_model.DR_SR_success_with_ack(M, N, k_now, Q, P_f, lamda, alpha)
    index = 1
    mark = ['k:x', 'k:D', 'k:o', 'k:^', 'k-.+']
    for row in arr_result:
        print(row)
        line, = plt.plot(X_points, row, mark[index - 1],
                         markersize=7, markerfacecolor='none')
        label = 'K=' + str(index)
        line.set_label(label)
        plt.legend()
        index = index + 1

    plt.xscale('log')
    plt.ylim([0, 1.1])
    plt.xlim([0.000001, 1])
    plt.xlabel("BLER of the 1st transmission of each packet ($P_{1}$)")
    plt.ylabel("Success probability")
    plt.savefig('s_prob_new_alpha_N.eps', dpi=5000)
    plt.show()

def plot_success_prob_BLER_k_improving_ratio(N,Q, M, lamda, alpha, P_f):
    # Get some results
    num_X = 6
    prob_success, K = (num_X, 5)
    arr_result = [[0 for i in range(prob_success)] for j in range(K)]

    result = []
    X_points = []
    for i in range(K):
        result.clear()
        P_f.clear()
        k_now = i + 1
        for j in range(num_X):
            P_f_1 = 1e-1 ** j
            if i==0:
                X_points.append(P_f_1)
            P_f = [P_f_1 * math.pow(1e-01, i) for i in range(K)]  # initial faiures
            arr_result[i][j] = GF_model.DR_SR_success_with_ack(M, N, k_now, Q, P_f, lamda, alpha)
    index = 1
    mark = ['k:x', 'k:D', 'k:o', 'k:^', 'k-.+']
    pre  = []
    for row in arr_result:
        print(row)

        # calculate
        y = []
        y.clear()
        if index == 1:
            base = row
            pre  = row
        else:
            for k in range(len(row)):
                if k != 0:
                    y.append((row[k]-pre[k])*100/row[k])
            line, = plt.plot(X_points[1:], y, mark[index - 1],
                 markersize=7, markerfacecolor='none')
            label = 'K=' + str(index)
            line.set_label(label)
            plt.legend()
            pre = row
        index = index + 1


    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([0, 100])
    plt.xlim([0.000005, 0.2])
    plt.xlabel("BLER of the 1st transmission of each packet ($P_{1}$)")
    plt.ylabel("Relative improvement \nof success probability (%)")
    plt.savefig('s_prob_new_alpha_N.eps', dpi=5000)
    plt.show()

def plot_success_prob_M_k(N,Q, lamda, alpha, P_f_1):
    # Get some results
    num_X = 6
    M, K = (num_X, 5)
    arr_result = [[0 for i in range(M)] for j in range(K)]

    result = []
    X_points = []
    for i in range(K):
        result.clear()
        k_now = i + 1
        for j in range(num_X):
            M_value = j + 1
            if i==0:
                X_points.append(M_value)
            P_f = [P_f_1 * math.pow(P_f_1, i) for i in range(K)]  # initial faiures
            arr_result[i][j] = GF_model.DR_SR_success_with_ack(M_value, N, k_now, Q, P_f, lamda, alpha)
        P_f.clear()

    index = 1
    mark = ['k:x', 'k:D', 'k:o', 'k:^', 'k-.+']

    #######################################################
    # Graph plotting
    fig, ax1 = plt.subplots()
    ax2 = fig.add_axes([0.26, 0.3, 0.3, 0.3])

    for row in arr_result:
        print(row)

        ### Original graph

        line, = ax1.plot(X_points, row, mark[index - 1],
                         markersize=7, markerfacecolor='none')
        line.set_label('K=' + str(index))
        ax1.legend()

        ### Inset graph
        ax2.plot(X_points, row, mark[index - 1], markersize=7, markerfacecolor='none')
        ###
        index = index + 1

    ax1.set_xlabel("Shared Resource (M)")
    ax1.set_ylabel("Success probability")
    ax1.set_xlim([1, num_X])
    ax1.set_ylim([0.89, 1])

    ax2.set_xlabel("M")
    #ax2.set_ylabel("S_prob")
    ax2.get_xaxis().set_visible(False)
    #ax2.get_yaxis().set_visible(False)
    ax2.set_xlim([3.95, 4.05])
    ax2.set_ylim([0.9878, 0.98931])

    rectangle = patches.Rectangle((3.9, 0.9835), 0.2, 0.009, fc='none', ec="black", linestyle="-.")
    ax1.add_patch(rectangle)

    line=plt.Line2D((1.87,3.9),(0.96,0.9926), lw=0.5, ls='-.', color='black')
    ax1.add_line(line)
    line= plt.Line2D((3.81, 4.11), (0.9171, 0.9832), lw=0.5, ls='-.', color='black')
    ax1.add_line(line)

    #plt.savefig('s_prob_new_alpha_N.eps', dpi=5000)
    plt.show()
    plt.close()

def plot_success_prob_alpha_k(N,Q, M, lamda, P_f_1):
    # Get some results
    num_X = 10
    alpha, K = (num_X, 5)
    arr_result = [[0 for i in range(alpha)] for j in range(K)]

    result = []
    X_points = []
    for i in range(K):
        result.clear()
        k_now = i + 1
        for j in range(num_X):
            alpha_value = 0.1*j + 0.1
            if i==0:
                X_points.append(alpha_value)
            P_f = [P_f_1 * math.pow(P_f_1, i) for i in range(k_now)]  # initial faiures
            arr_result[i][j] = GF_model.DR_SR_success_with_ack(M, N, k_now, Q, P_f, lamda, alpha_value)
        P_f.clear()

    index = 1
    mark = ['k:x', 'k:D', 'k:o', 'k:^', 'k-.+']

    #######################################################
    # Graph plotting
    fig, ax1 = plt.subplots()
    ax2 = fig.add_axes([0.3, 0.3, 0.3, 0.3])

    for row in arr_result:
        print(row)

        ### Original graph

        line, = ax1.plot(X_points, row, mark[index - 1],
                         markersize=7, markerfacecolor='none')
        line.set_label('K=' + str(index))
        ax1.legend()

        ### Inset graph
        ax2.plot(X_points, row, mark[index - 1], markersize=7, markerfacecolor='none')
        ###
        index = index + 1

    ax1.set_xlabel("alpha ($\\alpha$)")
    ax1.set_ylabel("Success probability")
    ax1.set_xlim([0.1, num_X*0.1])
    ax1.set_ylim([0.89, 1])

    ax2.set_xlabel("M")
    #ax2.set_ylabel("S_prob")
    ax2.get_xaxis().set_visible(False)
    #ax2.get_yaxis().set_visible(False)
    ax2.set_xlim([0.495, 0.505])
    ax2.set_ylim([0.9825, 0.984])

    rectangle = patches.Rectangle((0.485, 0.9785), 0.03, 0.009, fc='none', ec="black", linestyle="-.")
    ax1.add_patch(rectangle)

    line=plt.Line2D((0.302,0.485),(0.96,0.9782), lw=0.5, ls='-.', color='black')
    ax1.add_line(line)
    line= plt.Line2D((0.517, 0.6505), (0.9782, 0.9602), lw=0.5, ls='-.', color='black')
    ax1.add_line(line)

    #plt.savefig('s_prob_new_alpha_N.eps', dpi=5000)
    plt.show()
    plt.close()

def plot_with_without_success_prob_M_k(N,Q, lamda, alpha, P_f_1):
    # Get some results
    num_X = 6
    M, K = (num_X, 5)
    arr_result1 = [[0 for i in range(M)] for j in range(K)]
    arr_result2 = [[0 for i in range(M)] for j in range(K)]

    result = []
    X_points = []
    for i in range(K):
        result.clear()
        k_now = i + 1
        for j in range(num_X):
            M_value = j + 1
            if i==0:
                X_points.append(M_value)
            P_f = [P_f_1 * math.pow(P_f_1, i) for i in range(k_now)]  # initial faiures
            arr_result1[i][j] = GF_model.DR_SR_success_with_ack(M_value, N, k_now, Q, P_f, lamda, alpha)
            arr_result2[i][j] = GF_model.DR_SR_success_without_ack(M_value, N, k_now, Q, P_f, lamda, alpha)
        P_f.clear()

    index = 1
    mark = ['k:x', 'k:D', 'k:o', 'k:^', 'k-.+']

    #######################################################
    # Graph plotting
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()

    for row1, row2 in zip(arr_result1, arr_result2):
        if index ==  3:
            line1, = ax1.plot(X_points, row1, 'k:s', markersize=8, markerfacecolor='none')
            line2, = ax1.plot(X_points, row2, 'k-o', markersize=8, markerfacecolor='none')

            #line1, = ax1.plot(X_points, [1- i for i in row1], 'k:s', markersize=8, markerfacecolor='none')
            #line2, = ax1.plot(X_points, [1- i for i in row2], 'k-o', markersize=8, markerfacecolor='none')

            #ax2.set_ylabel('Delay probability')  # we already handled the x-label with ax1
            #ax2.plot(X_points, [1- i for i in row1], 'k-.x', markersize=8, markerfacecolor='none')
            #ax2.plot(X_points, [1 - i for i in row2], 'k-.o', markersize=8, markerfacecolor='none')

            line1.set_label('with early stopping' )
            line2.set_label('without early stopping')
            ax1.legend()
        index = index + 1

    ax1.set_xlabel("Shared Resource (M)")
    ax1.set_ylabel("Delay probability")
    ax1.set_xlim([1, num_X])
    ax1.set_ylim([0.9, 1.0])
    plt.savefig('with_without_dprob_M.eps', dpi=5000)
    plt.show()
    plt.close()

def plot_with_without_success_prob_BLER_k(N,Q, M, lamda, alpha, P_f):
    # Get some results
    num_X = 6
    prob_success, K = (num_X, 5)
    arr_result1 = [[0 for i in range(prob_success)] for j in range(K)]
    arr_result2 = [[0 for i in range(prob_success)] for j in range(K)]

    result = []
    X_points = []
    for i in range(K):
        result.clear()
        P_f.clear()
        k_now = i + 1
        for j in range(num_X):
            P_f_1 = 1e-1 ** j
            if i==0:
                X_points.append(P_f_1)
            P_f = [P_f_1 * math.pow(1e-01, i) for i in range(k_now)]  # initial faiures
            arr_result1[i][j] = GF_model.DR_SR_success_with_ack(M, N, k_now, Q, P_f, lamda, alpha)
            arr_result2[i][j] = GF_model.DR_SR_success_without_ack(M, N, k_now, Q, P_f, lamda, alpha)
    index = 1
    mark = ['k:x', 'k:D', 'k:o', 'k:^', 'k-.+']
    for row1, row2 in zip(arr_result1, arr_result2):
        if index == 3:
            #line1, = plt.plot(X_points, row1, 'k:s', markersize=9, markerfacecolor='none')
            #line2, = plt.plot(X_points, row2, 'k-o', markersize=9, markerfacecolor='none')
            line1, = plt.plot(X_points, [1- i for i in row1], 'k:s', markersize=9, markerfacecolor='none')
            line2, = plt.plot(X_points, [1- i for i in row2], 'k-o', markersize=9, markerfacecolor='none')
            line1.set_label('with early stopping')
            line2.set_label('without early stopping')
            plt.legend()
        index = index + 1

    plt.xscale('log')
    plt.ylim([0, 0.7])
    plt.xlim([0.00001, 1])
    plt.xlabel("BLER of the 1st transmission of each packet ($P_{1}$)")
    plt.ylabel("Delay probability")
    #plt.savefig('s_prob_new_alpha_N.eps', dpi=5000)
    plt.show()

def run_all_numerical():

    # Parameter spaces
    N = 12  # Number of UEs
    lamda = 0.5  # arrival rate (being activated rate)
    alpha = 1.0

    P_f_list = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
                0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    all_results = []
    for i in reversed(P_f_list):
        P_f_1 = i
        for K in range(1, 6, 1):
            P_f = [P_f_1 * math.pow(1e-01, j) for j in range(K)]  # initial failures

            for M in range(0, N, 1):

                for Q in range(1, 3, 1):

                    if K > Q + M:
                        continue
                    else:
                        without_R = GF_model.DR_SR_success_without_ack(M, N, K, Q, P_f, lamda, alpha)
                        with_R = GF_model.DR_SR_success_with_ack(M, N, K, Q, P_f, lamda, alpha)
                        # print("M:", M,
                        #                ", BLER: ", P_f_1,
                        #                ", without ack: ", "{:1.5f}".format(truncate(without_R, 5)),
                        #                "----- U_ratio: ", "{:1.3f}".format(truncate((N * Q + M) / (N * (M + Q)),3)))
                        # print("M:    ", M,
                        #                "BLER: ", P_f_1,
                        #                "with ack   : ", "{:1.5f}".format(truncate(with_R, 5)),
                        #                "----- U_ratio: ", "{:1.3f}".format(truncate((N * Q + M) / (N * (M + Q)),3)))
                        # if without_R > 0.99999:
                        print("Q:", Q,
                              "M:", M,
                              "K:", K,
                              "BLER: ", i,
                              "without ack: ", "{:1.10f}".format(truncate(without_R, 10)),
                              "----- U_ratio: ", "{:1.0f}".format(truncate((N * Q + M), 3)))
                        # if with_R > 0.99999:
                        print("Q:", Q,
                              "M:", M,
                              "K:", K,
                              "BLER: ", i,
                              "with ack   : ", "{:1.10f}".format(truncate(with_R, 10)),
                              "----- U_ratio: ", "{:1.0f}".format(truncate((N * Q + M), 3)))


def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ############################################################################
    # Parameter spaces
    N = 3  # Number of UEs
    Q = 1  # Number of dedicated resources, N1 = Q
    K = 3  # Number of trials,
    M = 5  # Amount of shared resources
    lamda = 0.3  # arrival rate (being activated rate)
    alpha = 1.0
    P_f_1 = 1e-1  # first BLER (BLOCK ERROR RATE) p_f_1 = 10*p_f_2 ....
    P_f = [P_f_1 * math.pow(1e-01, i) for i in range(K)] # initial faiures
    #############################################################################

    # 20210622
    #plot_success_prob_BLER_k(N, Q, M, lamda, alpha, P_f)
    #plot_success_prob_BLER_k_improving_ratio(N, Q, M, lamda, alpha, P_f)
    #plot_success_prob_M_k(N, Q, lamda, alpha, P_f_1)
    #plot_success_prob_alpha_k(N, Q, M, lamda, P_f_1)

    # 20210706
    #plot_with_without_success_prob_M_k(N, Q, lamda, alpha, P_f_1)
    #plot_with_without_success_prob_BLER_k(N, Q, M, lamda, alpha, P_f)
    #to find the amount of resources which satisfy success probability

    # 20210719
    # run_all_numerical

    N=32
    alpha = 1.0
    p_i=[]
    K = 10
    Q = 10
    M = 100
    P_f_list = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
            0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]

    with open('GF_result', 'a') as f:
        print('{2:>3},{3:>8},{4:>2},{5:>2},{6:>3},{7:>9},{8:>9}'
              .format("N", "aph", "lda", "BLER", "K", "Q", "M", "n_ack", "y_ack"), file=f)

    for lamda in range(10):
        l_i = lamda * 0.1 + 0.1
        for BLER in P_f_list:
            p_i.clear()
            p_i = [BLER * math.pow(1e-01, i) for i in range(K)]
            for k_i in range(1, K):
                for q_i in range(1, Q):
                    for m_i in range(1, M):
                        if q_i + m_i >= K:

                            ###############################################################################
                            # Multiprocessing
                            pool = multiprocessing.Pool(processes=)

                            result1 = GF_model.DR_SR_success_without_ack(m_i, N, k_i, q_i, p_i, l_i, alpha)
                            result2 = GF_model.DR_SR_success_with_ack(m_i, N, k_i, q_i, p_i, l_i, alpha)
                            if result1 >= 0.99999 or result2 >= 0.99999:
                                with open('GF_result','a') as f:
                                    print('{2:>3.1f},{3:>8.5f},{4:>2d},{5:>2d},{6:>3d},{7:>9.6f},{8:>9.6f}'
                                          .format(N, alpha, l_i, BLER, k_i, q_i, m_i, result1, result2), file=f)







"""
    P_f = []
    P_f = [0.001111, 0.002, 0.001]
    # DR_SR_success_without_ack(M, N, K, Q, P_f, lamda, alpha):
    result1 = GF_model.DR_SR_success_without_ack(1, 32, 3, 0, P_f, lamda, alpha)
    print(result1)

    P_f = []
    P_f = [0.001111, 0.002, 0.001, 0.0001, 0.0001]
    # DR_SR_success_with_ack(M, N, K, Q, P_f, lamda, alpha):
    result2 = GF_model.DR_SR_success_with_ack(1, 32, 5, 0, P_f, lamda, alpha)
    print(result2)
"""