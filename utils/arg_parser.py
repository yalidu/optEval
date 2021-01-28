import argparse


def parse_arg():

    # ! DO NOT USE action='store_false', SINCE FALSE FLAG WILL BE IGNORED IN dist_train.py
    parser = argparse.ArgumentParser()

    parser.add_argument('-game', type=str, default='traffic_junction')

    # common settings
    parser.add_argument('-sig_type', type=str, default='continuous')
    parser.add_argument('-sig_type_2', type=str, default='continuous')
    parser.add_argument('-sig_size', type=int, default=8)  # 4, 8, 16
    parser.add_argument('-sig_size_2', type=int, default=8)  # 4, 8, 16
    parser.add_argument('-sig_interval', type=int, default=4)  # 4, 10, 20
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-gamma_2', type=float, default=0.99)
    parser.add_argument('-lam', type=float, default=0.8)
    parser.add_argument('-hid_size', type=str, default='15,8')  # 15,8  20,10,5
    parser.add_argument('-policy_sizes', type=str, default='15,8')  # 15,8  20,10,5
    parser.add_argument('-critic_sizes', type=str, default='15,8')  # 15,8  20,10,5
    parser.add_argument('-u_sizes', type=str, default='15,8')  # 15,8  20,10,5
    parser.add_argument('-ac_fn', type=str, default='tanh')  # relu, tanh
    parser.add_argument('-opt_fn', type=str, default='adam')  # adam, adagrad
    parser.add_argument('-bs', type=int, default=1000)
    parser.add_argument('-bs_2', type=int, default=1000)
    parser.add_argument('-sampling_len', type=int, default=-1)
    parser.add_argument('-sampling_len_2', type=int, default=-1)
    parser.add_argument('-max_step', type=int, default=500)
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-lr_2', type=float, default=5e-3)
    parser.add_argument('-eps', type=float, default=0)
    parser.add_argument('-reg_coef', type=float, default=1,
                        help='neglogpac regularization coefficient')
    parser.add_argument('-vf_coef', type=float, default=0.5,
                        help='value function loss coefficient')
    parser.add_argument('-u_coef', type=float, default=0.5,
                        help='U function loss coefficient')
    parser.add_argument('-critic_coef', type=float, default=0.5,
                        help='Q function loss coefficient')
    parser.add_argument('-ent_coef', type=float, default=0,
                        help='entropy function loss coefficient')
    parser.add_argument('-deterministic', action='store_true', default=False)
    parser.add_argument('-max_agt_num', type=int, default=5)
    parser.add_argument('-alg', type=str, default='a2c')
    parser.add_argument('-pol', type=str, default='mlp')
    parser.add_argument('-model_name', type=str, default='mlp')
    parser.add_argument('-model_name_2', type=str, default=None)
    parser.add_argument('-net', type=str, default='mlp')
    parser.add_argument('-independent', type=int, default=0,
                        help='enabling independent policy & value networks setting. '
                             '0=disabled, 1=entry-based, 2=route-based')
    parser.add_argument('-one_player', action='store_true', default=False)
    parser.add_argument('-selfplay', action='store_true', default=False)
    parser.add_argument('-parameter_sharing', action='store_true', default=False)
    parser.add_argument('-no_rnn', action='store_true', default=False)
    parser.add_argument('-soft', action='store_true', default=False)
    parser.add_argument('-soft_coef', type=float, default=1e-2)
    parser.add_argument('-num_head', type=int, default=4)
    parser.add_argument('-target_critic_update_interval', type=int, default=-1)
    parser.add_argument('-delta', type=float, default=1e-2,
                        help='soft-update of target critic')

    # matrix_game settings
    parser.add_argument('-num_agt', type=int, default=2)
    parser.add_argument('-num_step', type=int, default=4)
    parser.add_argument('-eval_interval', type=int, default=1000)
    parser.add_argument('-eval_number', type=int, default=5000)
    parser.add_argument('-common_signal', action='store_true', default=False)

    # pp settings
    parser.add_argument('-max_replay_buffer_len', type=int, default=25000)
    parser.add_argument('-num_units', type=int, default=64)

    # cona settings
    parser.add_argument('-test_ep_interval', type=int, default=-1,
                        help='test every N chief epochs')
    parser.add_argument('-test_step', type=int, default=-1)

    # training settings
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-init_epoch_num', type=int, default=0)
    parser.add_argument('-max_epoch_num', type=int, default=int(1e6))
    parser.add_argument('-grad_clip_norm', type=float, default=10)
    parser.add_argument('-no_lr_decay', action='store_true', default=False)

    # distributed training settings
    parser.add_argument('-num_worker', type=int, default=1)
    parser.add_argument('-job_name', type=str, default='')
    parser.add_argument('-job_idx', type=str, default='0')
    parser.add_argument('-env_addr', type=str, default='localhost')
    parser.add_argument('-env_port', type=str, default='0')
    parser.add_argument('-port', type=str, default='0')

    # IO settings
    parser.add_argument('-gpu', type=str, default='-1')
    parser.add_argument('-output_path', type=str, default='./res/')
    parser.add_argument('-restore_path', type=str, default='')
    parser.add_argument('-restore_path_2', type=str, default='')
    parser.add_argument('-view', action='store_true', default=False)
    parser.add_argument('-eval', action='store_true', default=False)
    parser.add_argument('-note', type=str, default='test')

    args = parser.parse_args()

    if args.output_path[-4:] == 'res/':
        args.output_path += '_%s/' % args.game  # rebase to sub-directory

    return args
