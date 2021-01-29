import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# sns.set(style='white', color_codes=True)

# agent.size = 0.075 if agent.adversary else 0.05

path = '/Users/lhchen/nas/res/_pptest/test/'
file = 'points.txt'
sig_flag = False


def squared_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def filter(ps, c, r):
    squared_r = r ** 2
    new_ps = []
    for p in ps:
        if squared_dist(p, c) > squared_r:
            new_ps.append(p)
    return new_ps


def check_collistion(a, b, r_a, r_b):
    thr = (r_a + r_b) ** 2
    if squared_dist(a, b) > thr:
        return None
    else:
        ratio = r_a / (r_a + r_b)
        return [a[0] + (b[0]-a[0])*ratio, a[1] + (b[1]-a[1])*ratio]


if __name__ == '__main__':

    n = 0
    sigs = []
    p1s = []
    p2s = []
    q1s = []
    q2s = []
    x1s = []
    x2s = []
    r_p = 0.075
    r_q = 0.05


    with open(path+file, 'r') as f:
        for line in f:
            n += 1
            # if n > 1000:
            #     break
            raw_data = line.strip().split('\t')

            raw_ps = raw_data[1][1:-1].split(',')
            ps = list(map(float, raw_ps))
            p1 = ps[:2]
            p2 = ps[2:4]
            q1 = ps[4:6]
            q2 = ps[6:8]
            p1s.append(p1)
            p2s.append(p2)
            q1s.append(q1)
            q2s.append(q2)

            for q in (q1, q2):
                x = check_collistion(p1, q, r_p, r_q)
                if x:
                    x1s.append(x)
                x = check_collistion(p2, q, r_p, r_q)
                if x:
                    x2s.append(x)

            if sig_flag:
                raw_sig = raw_data[0][1:-1].split(',')
                sig = list(map(float, raw_sig))
                sigs.append(sig)

    # p1s = filter(p1s, [-0.3, 0.], 0.25)
    # p2s = filter(p2s, [0.3, 0.], 0.25)

    sigs = np.array(sigs)
    p1s = np.array(p1s)
    p2s = np.array(p2s)
    ps = np.concatenate([p1s, p2s], 0)
    q1s = np.array(q1s)
    q2s = np.array(q2s)
    qs = np.concatenate([q1s, q2s], 0)
    x1s = np.array(x1s)
    x2s = np.array(x2s)
    xs = np.concatenate([x1s, x2s], 0)
    print('There are %i points in total' % len(ps))
    print('There are %i collisions in total' % len(xs))

    # g = sns.jointplot(x=ps[:, 0], y=ps[:, 1], kind='kde', space=0, color='cyan', xlim=(-2, 2), ylim=(-2, 2))
    # plt.show()
    # g = sns.jointplot(x=qs[:, 0], y=qs[:, 1], kind='kde', space=0, color=[0.35, 0.85, 0.35], xlim=(-1, 1), ylim=(-1, 1))
    # plt.show()
    g = sns.jointplot(x=xs[:, 0], y=xs[:, 1], kind='kde', space=0, color='orange', xlim=(-1, 1), ylim=(-1, 1))
    plt.show()
    # g = sns.jointplot(x=q1s[:, 0], y=q1s[:, 1], kind='kde', space=0, color=[0.85, 0.35, 0.85], xlim=(-1, 1), ylim=(-1, 1))
    # plt.show()
    # g = sns.jointplot(x=q2s[:, 0], y=q2s[:, 1], kind='kde', space=0, color=[0.85, 0.85, 0.35], xlim=(-1, 1), ylim=(-1, 1))
    # plt.show()
