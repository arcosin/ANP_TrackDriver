import sys
import time
import torch
import socket
import pickle
import numpy as np
from PIL import Image

from sac import SACAgent
from sac import BasicBuffer

num_updates = 8

# Purely for testing purposes
def receive_dummy_buffer(image_size, n=100):
    replay_buffer = BasicBuffer(int(1e6))
    for _ in range(n):
        state = np.random.randint(255, size=image_size)
        action = [np.random.random() * 100, -60 + np.random.random() * 120]
        reward = np.random.random()
        next_state = np.random.randint(255, size=image_size)
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
    return replay_buffer

def listen(agent, batch_size, host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    while True:
        print("Listening for buffer...")
        conn, addr = s.accept()
        print("Connected to %s\n" % str(addr))

        data = []
        start = time.time()
        while True:
            packet = conn.recv(1024)
            if not packet: break
            data.append(packet)
        replay_buffer = pickle.loads(b"".join(data))
        end = time.time()
        print("Received replay buffer from bot in %fs" % (end - start))
        print("Closing connection...\n")
        conn.close()

        print("Updating models...")
        agent.replay_buffer.add_to_buffer(replay_buffer)
        start = time.time()

        for i in range(num_updates):
            s2 = time.time()
            fe_model, pi_model, completed = agent.update(batch_size)
            e2 = time.time()

            if completed == False:
                print(f"\tNot enough experiences in replay buffer, terminating update sequence")
                break

            t_time = s2 - e2
            print(f"\tFinished update {i} in {t_time}")
        
        end = time.time()
        print("Finished all updates, total time: %fs\n" % (end - start))

        print("Pickling models...")
        start = time.time()
        models = pickle.dumps((fe_model, pi_model))
        end = time.time()
        print("Pickled models in %fs\n" % (end - start))

        # NOTE: I was not able to get this to work with the same connection,
        #       so I close it and make a new one. It might be possible to use the same connection.
        conn, addr = s.accept()
        print("Sending updated models...")
        start = time.time()
        conn.sendall(models)
        end = time.time()
        print("Models sent to bot in %fs" % (end - start))
        print("Closing connection...\n")
        conn.close()


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
    parser.add_option('-B', '--batch', dest='batch_size', type='int',
                      help=default('batch size'), default=32)

    parser.add_option('--checkpoint', dest='checkpoint_path',
                      help=default('Path to saved checkpoint. Must contain:\n\
                                    fe, v_net, tv_net, q1_net, q2_net, pi_net\n\
                                    v_opt, q1_opt, q2_opt, pi_opt'), default=None)

    parser.add_option('--host', dest='host',
                      help=default('server hostname'), default='localhost')
    parser.add_option('--port', dest='port',type='int',
                      help=default('port number'), default=1138)

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

    args['host'] = options.host
    args['port'] = options.port

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

    listen(agent, args['batch_size'], args['host'], args['port'])
