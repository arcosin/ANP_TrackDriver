import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from sac import SACAgent

# Purely for testing purposes
def receive_dummy_buffer(image_size, n=100):
    from replay_buffers import BasicBuffer
    replay_buffer = BasicBuffer(int(1e6))
    for _ in range(n):
        state = np.random.randint(255, size=image_size)
        action = [np.random.random() * 100, -60 + np.random.random() * 120]
        reward = np.random.random()
        next_state = np.random.randint(255, size=image_size)
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
    return replay_buffer

def listen(agent, batch_size):
    # NOTE: This is PSEUDOCODE, implementation relies on the IPC API
    while True:
        # Listen for robot_controller buffer transmission
        # print("listening for buffer...")

        # replay_buffer  = ipc.receive_buffer()
        # print("received buffer")

        # print("updating models...")
        # fe_weights, pi_weights = agent.update(batch_size)

        # print("sending models...")
        # ipc.send_weights(fe_model, pi_model)
        print('listening for buffer...')
        input() # simulate waiting
        replay_buffer = receive_dummy_buffer((256, 256, 3))
        agent.set_replay_buffer(replay_buffer)

        print('updating models...')
        fe_model, pi_model = agent.update(batch_size)

        print('sending updated models...')
        print(fe_model)
        print()
        print(pi_model)

def readCommand(argv):
    def default(s):
        return s + ' [Default: %default]'

    from optparse import OptionParser
    usageStr = """
        PURPOSE:    Begin server to transmit updated policy network weights to bot
        USAGE:      python server.py <options>
    """
    parser = OptionParser(usageStr)

    parser.add_option('-t', '--tau', dest='tau', type='float',
                      help=default('tau'), default=0.005)
    parser.add_option('-g', '--gamma', dest='gamma', type='float',
                      help=default('gamma, discount factor'), default=0.99)
    parser.add_option('-a', '--alpha', dest='alpha', type='float',
                      help=default('alpha, value learning rate'), default=3e-3)
    parser.add_option('-b', '--beta', dest='beta', type='float',
                      help=default('beta, q learning rate'), default=3e-3)
    parser.add_option('-e', '--eta', dest='eta', type='float',
                      help=default('eta, policy learning rate'), default=3e-3)
    parser.add_option('-s', '--size', dest='size', type='int',
                      help=default('image dimension'), default=256)

    parser.add_option('--checkpoint', dest='checkpoint_path',
                      help=default('Path to saved checkpoint. Must contain:\n\
                                    fe, v_net, tv_net, q1_net, q2_net, pi_net\n\
                                    v_opt, q1_opt, q2_opt, pi_opt'), default=None)

    parser.add_option('-B', '--batch', dest='batch_size', type='int',
                      help=default('batch size'), default=32)

    options, junk = parser.parse_args(argv)
    if len(junk) != 0:
        raise Exception('Command line input not understood: ' + str(junk))
    args = dict()
    args['tau'] = options.tau
    args['gamma'] = options.gamma
    args['alpha'] = options.alpha
    args['beta'] = options.beta
    args['eta'] = options.eta
    args['size'] = (options.size, options.size, 3)
    args['batch_size'] = options.batch_size
    args['checkpoint_path'] = options.checkpoint_path

    return args

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = readCommand(sys.argv[1:])
    agent = SACAgent(action_range=[[0, 100], [-60, 60]],
                     action_dim=2,
                     gamma=args['gamma'],
                     tau=args['tau'],
                     v_lr=args['alpha'],
                     q_lr=args['beta'],
                     pi_lr=args['eta'],
                     image_size=args['size'],
                     kernel_size=(3,3),
                     conv_channels=4,
                     checkpoint=None if args['checkpoint_path'] == None else torch.load(args['checkpoint_path']))

    listen(agent, args['batch_size'])

