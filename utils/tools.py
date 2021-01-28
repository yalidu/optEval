import collections
import numpy as np


def add_nt(nts, d: dict):
    list_flag = True if type(nts) == list else False
    for k in d.keys():
        vs = d[k]
        if list_flag:
            assert len(nts) == len(vs)
            for i in range(len(nts)):  # nt = namedtuple
                t = getattr(nts[i], k)  # t = target
                if type(t) == list:
                    t.append(vs[i])
                else:
                    raise NotImplementedError
        else:
            t = getattr(nts, k)
            if type(t) == list:
                t.append(vs)
            else:
                raise NotImplementedError


def compute_bound(n_in, n_out):
    return np.sqrt(6/(n_in + n_out))


def plot_in_terminal(input, slot_num=10, length=50, mode='static'):
    assert mode in ('static', 'dynamic')
    if len(input) > length:
        input = input[-length:]
    input = np.array(input)
    assert len(input.shape) == 1
    if len(input):
        _plot_in_terminal(np.array(input), slot_num, mode)
    else:
        print('[Warning] Nothing to plot in terminal.')


def _plot_in_terminal(input, slot_num, mode):
    max_val = np.max(input)
    min_val = np.min(input)
    if mode == 'dynamic':
        if max_val == min_val:
            if max_val > 0:
                min_val = max_val / slot_num
            elif max_val == 0:
                min_val = -1
            else:
                max_val = min_val / slot_num
    elif mode == 'static':  # the height of y=0 is fixed
        abs_val = max(abs(max_val), abs(min_val))
        if abs_val == 0:
            abs_val = 1
        max_val = 0 + abs_val
        min_val = 0 - abs_val

    slot_size = (max_val - min_val) / slot_num

    res = np.reshape([' ']*(slot_num+1)*len(input), (slot_num+1, len(input)))
    res[:] = ' '
    for i in range(len(input)):
        elem = input[i]
        slot_idx = np.int(slot_num - (elem - min_val) // slot_size)
        res[slot_idx, i] = '.'

    if max_val > 0 and min_val > 0:
        print('-' * (10 + len(input)) + '|')

    for i in range(slot_num+1):
        header = '%.6f' % (max_val - i * slot_size)
        line = res[i]
        print(header[:10] + ' '*(10-len(header)) + ''.join(list(line))+'|')
        if max_val > 0 and min_val < 0 and \
                max_val - i * slot_size >= 0 and max_val - (i+1) * slot_size < 0:
            print('-' * (10 + len(input)) + '|')

    if max_val < 0 and min_val < 0:
        print('-' * (10 + len(input)) + '|')


# _plot_in_terminal([-1, 1, -2, 2, 3, 4, 5, -10], 20, mode='static')