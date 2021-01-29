import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.manifold import TSNE


def parse_arg():

    parser = argparse.ArgumentParser()

    parser.add_argument('-input_path', type=str, default='/home/lhchen/nas/res/_matrix_game/')
    parser.add_argument('-output_path', type=str, default='/home/lhchen/nas/res/_matrix_game/')
    parser.add_argument('-note', type=str, default='0')

    args = parser.parse_args()

    return args


def read_train_line(line):
    raw_data = line.split(' ')
    return float(raw_data[1])


def read_eval_block(block):

    if not block:
        return []
    sig_row_list = []
    sig_col_list = []
    ac_row_list = []
    ac_col_list = []
    for line in block:
        ep, sig_row, sig_col, ac_row, ac_col = read_eval_line(line)
        sig_row_list.append(sig_row)
        sig_col_list.append(sig_col)
        ac_row_list.append(ac_row)
        ac_col_list.append(ac_col)
    return ep, sig_row_list, sig_col_list, ac_row_list, ac_col_list


def read_eval_line(line):
    raw_data = line.split('|')
    raw_data[-1] = raw_data[-1].split(' ')[0][:-1]
    # print(raw_data)
    ep = int(raw_data[0][6:][:-1])
    sig_row = []
    sig_col = []
    temp = raw_data[1][1:][:-1]
    temp = temp.strip().split(', ')
    for elem in temp:
        sig_row.append(float(elem.strip()))
    temp = raw_data[2][1:][:-1]
    temp = temp.strip().split(', ')
    for elem in temp:
        sig_col.append(float(elem.strip()))
    # print(raw_data)
    ac_row = int(raw_data[3])
    ac_col = int(raw_data[4])

    return ep, sig_row, sig_col, ac_row, ac_col


if __name__ == '__main__':

    args = parse_arg()
    input_path = args.input_path + args.note + '/'
    output_path = args.output_path + args.note + '/'

    train_curve = []
    eval_block = []
    tsne = TSNE(n_components=2, random_state=0)

    def visualize(data, acts, save_path):
        if not list(np.array(data).shape)[1] == 2:
            trans_data = tsne.fit_transform(np.array(data))
        else:
            trans_data = np.array(data)
        plt.figure(figsize=(6, 5))
        colors = ['r', 'g', 'b', 'c']
        for i in range(len(data)):
            plt.scatter(trans_data[i, 0], trans_data[i, 1], c=colors[acts[i]], label=acts[i])
        portion = [acts.count(i)/len(acts) for i in [0, 1, 2, 3]]
        plt.title('0: %.3f, 1: %.3f, 2: %.3f, 3: %.3f' % (portion[0], portion[1], portion[2], portion[3]))
        plt.savefig(save_path)
        plt.cla()
        return portion

    with open(input_path + 'results.txt', 'r', encoding='utf8') as f:
        count = 0
        for line in f:
            if 'Train' in line:
                train_curve.append(read_train_line(line))
                if count:
                    eval_data = read_eval_block(eval_block)
                    if eval_data:
                        ep, sig_row, sig_col, ac_row, ac_col = eval_data
                        print(ep)
                        # print(sig_col[:10])
                        # print(ac_col[:10])
                        ac_row_portion = visualize(sig_row, ac_row, output_path+'row_%i.jpg' % ep)
                        ac_col_portion = visualize(sig_col, ac_col, output_path+'col_%i.jpg' % ep)
                    count = 0
                    eval_block = []
            if 'Eval' in line:
                eval_block.append(line)
                count += 1
            if len(train_curve) and not len(train_curve) % 500:
                plt.plot(range(len(train_curve)), np.array(train_curve))
                plt.savefig(output_path + 'train_curve.jpg')
                plt.cla()

    plt.plot(range(len(train_curve)), np.array(train_curve))
    plt.savefig(output_path+'train_curve.jpg')
    plt.cla()