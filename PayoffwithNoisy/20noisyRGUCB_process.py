import numpy as np
import pickle
for i in range(10, 201, 10):
    # np.save('./convnoisy/optm{}.npy'.format(players), convat)
    m = np.load('./convnoisy/optm{}.npy'.format(i))
    print(m)

for i in range(10, 201, 10):
    merror = 0
    pierror = 0
    for j in range(3):
        # print(os.path.getsize("/data/yanxue/alphaRankRunsconv/RGUCB{}_{}".format(i,j)))
        with open('/data/yanxue/alphaRankRunsconv/RGUCB{}_{}.pkl'.format(i, j), "rb") as f:
            data = pickle.load(f)
        merror += data["merror"]
        pierror += data["alpha_error"]
    merror /= 3
    pierror /= 3
    np.save("./convnoisy/{}.npy".format(i), [merror, pierror])
    print([merror, pierror])