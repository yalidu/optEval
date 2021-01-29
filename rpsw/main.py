import os
if 'http_proxy' in os.environ:
    os.environ.pop('http_proxy')
if 'https_proxy' in os.environ:
    os.environ.pop('https_proxy')
from utils.arg_parser import parse_arg
import socket
from time import gmtime, strftime


def is_available(ip, port):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        s.bind((ip, port))
        s.close()
        return True
    except:
        s.close()
        return False


def avaliable_ports(num_workers, env_addr):
    port = 12355
    if env_addr == 'localhost':
        host = '127.0.0.1'
    else:
        host = env_addr
    while not is_available(host, port):
        port += 1
    avaliable_ports = str(port)
    port += 1
    for _ in range(num_workers):
        while not is_available(host, port):
            port += 1
        avaliable_ports = avaliable_ports + ',' + str(port)
        port += 1
    return avaliable_ports


def avaliable_env_ports(num_workers, env_addr):
    port = 21330
    if env_addr == 'localhost':
        host = '127.0.0.1'
    else:
        host = env_addr
    while not is_available(host, port):
        port += 5
    avaliable_ports = [port]
    port += 5
    for _ in range(num_workers):
        while not is_available(host, port):
            port += 5
        avaliable_ports.append(port)
        port += 5
    return avaliable_ports


def change_to_comm(arg_dict, path):
    comm = ''
    for k in arg_dict.keys():
        v = arg_dict[k]
        if not v == '':
            if type(v) == bool:
                if v:
                    comm += '-%s ' % str(k)
            else:
                comm += '-%s %s ' % (str(k), str(v))
    # comm += ' > %s%s.%s.out 2>&1 & echo kill $! >>%skill.sh ' \
    #         % (path, arg_dict['job_name'], arg_dict['job_idx'],
    #            path)
    comm += ' & echo kill $! >>%skill.sh ' \
            % (path)
    return comm


if __name__ == '__main__':
    args = parse_arg()

    if args.num_worker < 0:

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        if args.game == 'traffic_junction':
            from snow.traffic_junction.traffic_junction import TrafficJunction as Game
        elif args.game == 'matrix_game':
            from snow.matrix_game.matrix_game import MatrixGame as Game
        elif args.game == 'combat':
            from snow.combat.combat import Combat as Game
            # python3 main.py -game combat
        else:
            raise NotImplementedError

        game = Game(args)
        game.run()
    else:
        num_worker = args.num_worker
        arg_dict = {k: v for k, v in args._get_kwargs()}
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        path = '%s%s/' % (args.output_path, args.note)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            filelist = [f for f in os.listdir(path)]
            for f in filelist:
                if (  # rerun the game.
                        args.restore_path == '' and
                        not f.startswith('.')
                ) or (  # continue the game in place
                        f.startswith('events.out.tfevents.') or
                        f == 'kill.sh' or
                        f == '.finish'
                        ):
                    os.remove(path + f)
        with open(path + 'results.txt', 'w') as f:
            f.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            f.write(str(args))

        comms = []
        comm = "CUDA_VISIBLE_DEVICES='-1' python3 %s/dist_train.py " % (os.path.dirname(os.path.abspath(__file__)))
        port = avaliable_ports(num_worker, args.env_addr)
        arg_dict['job_name'] = 'ps'
        arg_dict['port'] = str(port)
        comms.append(comm + change_to_comm(arg_dict, path))
        env_ports = avaliable_env_ports(num_worker, args.env_addr)
        arg_dict['job_name'] = 'worker'
        arg_dict['job_idx'] = '0'
        arg_dict['port'] = str(port)
        arg_dict['env_port'] = str(env_ports[-1])
        comms.append(comm + change_to_comm(arg_dict, path))
        for i in range(num_worker-1):
            arg_dict['job_idx'] = str(i+1)
            arg_dict['env_port'] = str(env_ports[i])
            comms.append(comm + change_to_comm(arg_dict, path))
        final_comm = ' &\n '.join(comms)

        # print(final_comm)
        os.system(final_comm)

        while True:
            filelist = [f for f in os.listdir(path)]
            if '.finish' in filelist:
                print('%s finished' % arg_dict['note'])
                break


