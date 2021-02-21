import pickle
import numpy as np
import os

def get_payoffs_bernoulli_game(n):
    U = np.random.uniform(-1, 1, size=(n, 5))
    V = np.random.uniform(-1, 1, size=(5, n))
    M = np.dot(U, V)
    print(np.linalg.matrix_rank(M))
    #print(M)
    '''
    M[:, :] = 0.5 * (M[:, :] - M[:, :].T)
    #
    print(M)
    

    # print(Mb)
    M_max = np.max(M)
    M_min = np.min(M)

    P = (M - M_min) / (M_max - M_min)

    M=2*P-1
    print(np.linalg.matrix_rank(P))
    print(P)
    # print(P)
    '''
    return M


def get_payoffs_bernoulli_gameP(size=(100, 10)):
    U = np.random.uniform(-1, 1, size=(1000, 5))
    V = np.random.uniform(-1, 1, size=(5, 1000))
    M = np.dot(U, V)
    M[:, :] = 0.1* (M[:, :] - M[:, :].T)

    # print(Mb)
    M_max = np.max(M)
    M_min = np.min(M)

    P = (M - M_min) / (M_max - M_min)
    # print(P)

    return M

if __name__ == '__main__':
    #M = get_payoffs_bernoulli_game()
    # print(M)
    #np.save("bernoulli100.npy",M)
    #np.save("gaussian15.npy", M)
    #M=np.load("gaussian15.npy")

    for i in range(10,201,10):
        M = get_payoffs_bernoulli_game(i)
        np.save("conv{}.npy".format(i), M)
    '''
    mmin=100000000
    mmax=-100000000
    for i in range(10,201,10):
        M=np.load("conv{}.npy".format(i))
        mmin=min(np.min(M),mmin)
        mmax=max(np.max(M),mmax)
        print(np.min(M),np.max(M))
    print(mmin)
    print(mmax)
    '''

    for i in range(10, 201, 10):
        #np.save('./convnoisy/optm{}.npy'.format(players), convat)
        m=np.load('./convnoisy/optm{}.npy'.format(i))
        print(m)

    for i in range(10,201,10):
        merror=0
        pierror=0
        for j in range(3):
            #print(os.path.getsize("/data/yanxue/alphaRankRunsconv/RGUCB{}_{}".format(i,j)))
            with open('/data/yanxue/alphaRankRunsconv/RGUCB{}_{}.pkl'.format(i,j),"rb") as f:
                data=pickle.load(f)
            merror+=data["merror"]
            pierror+=data["alpha_error"]
        merror/=3
        pierror/=3
        np.save("./convnoisy/{}.npy".format(i),[merror,pierror])
        print([merror,pierror])
