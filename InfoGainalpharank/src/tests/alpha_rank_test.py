import numpy as np
import sys
sys.path.append(".")
from src.alpha_rank import alpha_rank

def rock_paper_scissors(graphing=False):

    payoffs = np.array([[ 0, -1,  1],
                        [ 1,  0, -1],
                        [-1,  1,  0]])

    alphas = []
    strat_probs = []
    for alpha in np.logspace(-4, 2, num=60, base=10):
        phi = alpha_rank([payoffs], alpha=alpha)
        # print("Alpha: {:.2f}".format(alpha))
        assert np.all(np.isclose(np.array([1/3, 1/3, 1/3]), np.array(phi), atol=1e-4))
        alphas.append(alpha)
        strat_probs.append(tuple(phi))

    if graphing:
        import matplotlib.pyplot as plt

        action_names = ["Rock", "Paper", "Scissors"]
        for idx, name in enumerate(action_names):
            ys = [s[idx] for s in strat_probs]
            plt.plot(alphas, ys, label=name)
        plt.legend()
        plt.xscale("log")
        plt.ylim(0,1)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"Mass in $\pi$")
        plt.yticks([i/10 for i in range(11)])
        plt.grid(True, which="major")
        plt.savefig("graphs/alpha_rank_reprod/rps_alpha_strats.png")
        plt.close()

def biased_rock_paper_scissors_multipop(graphing=False, mutations=False):

    payoffs = np.array([
        [[   0, -0.5,   1],
        [ 0.5,    0, -0.1],
        [  -1,  0.1,    0]],
        [[   0, -0.5,   1],
        [ 0.5,    0, -0.1],
        [  -1,  0.1,    0]],
    ])


    mutations_list = [1,10,20,30,40,50,100] if mutations else [50]
    for m in mutations_list:
        if mutations:
            print("Mutation {}".format(m))
        alphas = []
        strat_probs = []
        for alpha in np.logspace(-4, 2, num=60, base=10):
            try:
                phi = alpha_rank(payoffs, alpha=alpha, mutation=m)
                alphas.append(alpha)
                strat_probs.append(tuple(phi))
            except ValueError:
                pass

        if graphing:
            import matplotlib.pyplot as plt

            action_names = ["{}_{}".format(a, b) for a in ["R", "P", "S"] for b in ["R", "P", "S"]]
            # action_names = ["Rock", "Paper", "Scissors"]
            for idx, name in enumerate(action_names):
                ys = [s[idx] for s in strat_probs]
                plt.plot(alphas, ys, label=name)
            plt.legend()
            plt.xscale("log")
            plt.ylim(0,1)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"Mass in $\pi$")
            plt.yticks([i/10 for i in range(11)])
            plt.grid(True, which="major")
            plt.savefig("graphs/alpha_rank_reprod/multipop_biased_rps_alpha_strats_{}m.png".format(m))
            plt.close()



def biased_rock_paper_scissors(graphing=False, mutations=False):

    payoffs = np.array([[   0, -0.5,    1],
                        [ 0.5,    0, -0.1],
                        [  -1,  0.1,    0]])


    mutations_list = [1,10,20,30,40,50,100] if mutations else [50]
    for m in mutations_list:
        if mutations:
            print("Mutation {}".format(m))
        alphas = []
        strat_probs = []
        for alpha in np.logspace(-4, 2, num=60, base=10):
            phi = alpha_rank([payoffs], alpha=alpha, mutation=m)
            # print("Alpha: {:.2f}".format(alpha))
            # probs = _get_rps_strat(phi)
            alphas.append(alpha)
            strat_probs.append(tuple(phi))

        if graphing:
            import matplotlib.pyplot as plt

            # action_names = ["{}_{}".format(a, b) for a in ["R", "P", "S"] for b in ["R", "P", "S"]]
            action_names = ["Rock", "Paper", "Scissors"]
            for idx, name in enumerate(action_names):
                ys = [s[idx] for s in strat_probs]
                plt.plot(alphas, ys, label=name)
            plt.legend()
            plt.xscale("log")
            plt.ylim(0,1)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"Mass in $\pi$")
            plt.yticks([i/10 for i in range(11)])
            plt.grid(True, which="major")
            plt.savefig("graphs/alpha_rank_reprod/biased_rps_alpha_strats_{}m.png".format(m))
            plt.close()

def bos(graphing=False):

    payoffs = [
        np.array([[3, 0],
                  [0, 2]]),
        np.array([[2, 0],
                  [0, 3]])
    ]

    alphas = []
    strat_probs = []
    # Goes haywire after 10^(-1)
    for alpha in np.logspace(-4, -1, num=30, base=10): 
        phi = alpha_rank(payoffs, alpha=alpha)
        alphas.append(alpha)
        strat_probs.append(tuple(phi))

    if graphing:
        import matplotlib.pyplot as plt

        action_names = ["OO", "OM", "MO", "MM"]
        for idx, name in enumerate(action_names):
            ys = [s[idx] for s in strat_probs]
            plt.plot(alphas, ys, label=name)
        plt.legend()
        plt.xscale("log")
        plt.ylim(0,1)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"Mass in $\pi$")
        plt.yticks([i/10 for i in range(11)])
        plt.grid(True, which="major")
        plt.savefig("graphs/alpha_rank_reprod/bos.png")
        plt.close()

def transpose_test(graphing=False):

    payoffs = [
        np.array([[ 0, 1],
                  [ 0, 0]]),
        np.array([[ 0, 1],
                  [ 0, 0]])
    ]

    alphas = []
    strat_probs = []
    for alpha in np.logspace(-4, 2, num=60, base=10):
        phi = alpha_rank(payoffs, alpha=alpha)
        alphas.append(alpha)
        strat_probs.append(tuple(phi))

    if graphing:
        import matplotlib.pyplot as plt

        action_names = ["AA", "AB", "BA", "BB"]
        for idx, name in enumerate(action_names):
            ys = [s[idx] for s in strat_probs]
            plt.plot(alphas, ys, label=name)
        plt.legend()
        plt.xscale("log")
        plt.ylim(-0.1,1.1)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"Mass in $\pi$")
        plt.yticks([i/10 for i in range(11)])
        plt.grid(True, which="major")
        plt.savefig("graphs/alpha_rank_reprod/transpose.png")
        plt.close()

def prisoners_dilemma(graphing=False):

    payoffs = [
        np.array([[-1, -3],
                  [ 0, -2]]),
        np.array([[ -1, 0],
                  [ -3, -2]])
    ]

    alphas = []
    strat_probs = []
    for alpha in np.logspace(-4, 2, num=60, base=10):
        phi = alpha_rank(payoffs, alpha=alpha)
        alphas.append(alpha)
        strat_probs.append(tuple(phi))

    if graphing:
        import matplotlib.pyplot as plt

        action_names = ["CC", "CD", "DC", "DD"]
        for idx, name in enumerate(action_names):
            ys = [s[idx] for s in strat_probs]
            plt.plot(alphas, ys, label=name)
        plt.legend()
        plt.xscale("log")
        plt.ylim(-0.1,1.1)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"Mass in $\pi$")
        plt.yticks([i/10 for i in range(11)])
        plt.grid(True, which="major")
        plt.savefig("graphs/alpha_rank_reprod/prisoners_dilemma.png")
        plt.close()
        

def bernoulli_one_pop(graphing=False):

    payoffs = [
         np.array(
             [[0.5,  0.62, 0.3 ],
              [0.26, 0.33, 0.9 ],
              [0.36, 0.01, 0.94]])
    ]

    alphas = []
    strat_probs = []
    for alpha in np.logspace(-4, 2, num=60, base=10):
        phi = alpha_rank(payoffs, alpha=alpha)
        alphas.append(alpha)
        strat_probs.append(tuple(phi))

    if graphing:
        import matplotlib.pyplot as plt

        action_names = ["A", "B", "C"]
        for idx, name in enumerate(action_names):
            ys = [s[idx] for s in strat_probs]
            plt.plot(alphas, ys, label=name)
        plt.legend()
        plt.xscale("log")
        plt.ylim(-0.1,1.1)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"Mass in $\pi$")
        plt.yticks([i/10 for i in range(11)])
        plt.grid(True, which="major")
        plt.savefig("graphs/alpha_rank_reprod/bernoulli_one_pop.png")
        plt.close()

def infinite_alpha_rank_tests():
    # Rock paper scissors
    payoffs = np.array([[ 0, -1,  1],
                        [ 1,  0, -1],
                        [-1,  1,  0]])
    phi = alpha_rank([payoffs], use_inf_alpha=True, use_sparse=False)
    np.testing.assert_almost_equal(np.array([1/3,1/3,1/3]), phi)
    phi = alpha_rank([payoffs], use_inf_alpha=True, use_sparse=True)
    np.testing.assert_almost_equal(np.array([1/3,1/3,1/3]), phi)
    
    # Biased rock paper scissors
    payoffs = np.array([[   0, -0.5,    1],
                        [ 0.5,    0, -0.1],
                        [  -1,  0.1,    0]])
    phi = alpha_rank([payoffs], use_inf_alpha=True, use_sparse=False)
    np.testing.assert_almost_equal(np.array([1/3,1/3,1/3]), phi)
    phi = alpha_rank([payoffs], use_inf_alpha=True, use_sparse=True)
    np.testing.assert_almost_equal(np.array([1/3,1/3,1/3]), phi)
    
    # Battle of sexes
    payoffs = [
        np.array([[3, 0],
                  [0, 2]]),
        np.array([[2, 0],
                  [0, 3]])
    ]
    phi = alpha_rank(payoffs, use_inf_alpha=True, inf_alpha_eps=0.0000001, use_sparse=False) # Need a small eps to ensure its close for this small task
    np.testing.assert_almost_equal(np.array([1/2, 0, 0, 1/2]), phi)
    phi = alpha_rank(payoffs, use_inf_alpha=True, inf_alpha_eps=0.0000001, use_sparse=True) # Need a small eps to ensure its close for this small task
    np.testing.assert_almost_equal(np.array([1/2, 0, 0, 1/2]), phi)
    
    payoffs = [
         np.array(
             [[0.5,  0.62, 0.3 ],
              [0.26, 0.33, 0.9 ],
              [0.36, 0.01, 0.94]])
    ]
    phi = alpha_rank(payoffs, use_inf_alpha=True, inf_alpha_eps=0.0000001, use_sparse=False) # Need a small eps to ensure its close for this small task
    np.testing.assert_almost_equal(np.array([1/3, 1/3, 1/3]), phi)
    phi = alpha_rank(payoffs, use_inf_alpha=True, inf_alpha_eps=0.0000001, use_sparse=True) # Need a small eps to ensure its close for this small task
    np.testing.assert_almost_equal(np.array([1/3, 1/3, 1/3]), phi)
    
    
# Testing the implementation of alpha rank
if __name__ == "__main__":
    # Finite alpha tests, mostly done by eye (comparing graphs to the alpha rank paper, https://arxiv.org/abs/1903.01373)
    graphing = True
    transpose_test(graphing=graphing)
    rock_paper_scissors(graphing=graphing)
    biased_rock_paper_scissors(graphing=graphing, mutations=False)
    biased_rock_paper_scissors_multipop(graphing=graphing, mutations=False)
    bos(graphing=graphing)
    prisoners_dilemma(graphing=graphing)
    bernoulli_one_pop(graphing=graphing)

    # Infinite alpha tests
    infinite_alpha_rank_tests()