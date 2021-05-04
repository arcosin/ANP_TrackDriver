import sys
import time
import torch
import socket
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from os import path

# sys.path.append(path.join(path.dirname(__file__), '..'))
from sac import SACAgent
from sac import BasicBuffer
from sac import Checkpointer

num_updates = 20
episode_rewards = []

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

    episode_num = 0

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
        data = pickle.loads(b"".join(data))
        replay_buffer = data["replay_buf"]
        episode_rewards.append(data["episode_reward"])
        end = time.time()
        print("Received replay buffer and episode_rewards from bot in %fs" % (end - start))
        print("Closing connection...\n")
        conn.close()

        checkpoint_num = input("Enter y to save current model, or press enter to skip and continue: ")
        if checkpoint_num == "y":
            print("Saving checkpoint")
            chkpt.save_checkpoint(agent)

        # Save some images
        print("Saving images...")
        save_episode_pictures(replay_buffer, episode_num, 10)
        episode_num += 1

        print("Updating models...")
        agent.replay_buffer.add_to_buffer(replay_buffer)
        start = time.time()

        for i in range(num_updates):
            s2 = time.time()
            fe_model, pi_model, completed, losses = agent.update(batch_size)
            e2 = time.time()

            if completed == False:
                print(f"\tNot enough experiences in replay buffer, terminating update sequence")
                break

            t_time = e2 - s2
            print(f"\tFinished update {i} in {t_time}")
        
        end = time.time()
        print("Finished all updates, total time: %fs\n" % (end - start))

        print("Updating reward graph...")
        start = time.time()
        update_reward_graph(episode_rewards)
        end = time.time()
        print("Updated reward graph in %fs\n" % (end - start))

        if losses is not None:
            print("Updating loss graph...")
            start = time.time()
            update_loss_graph(losses)
            end = time.time()
            print("Updated loss graph in %fs\n" % (end - start))

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

# Give sample=-1 to save all images
def save_episode_pictures(replay_buf, episode_num, sample):
    import random
    if sample > len(replay_buf): sample = len(replay_buf)
    if sample > 0:
        batch = random.sample(replay_buf, sample)
    else:
        batch = replay_buf
    i = 0
    for (state, _, _, _, _) in batch:
        im = Image.fromarray(state).rotate(90,expand=True)
        savepath = PIC_DIR + "img" + str(episode_num) + '-' + str(i) + ".jpg"
        im.save(savepath)
        i += 1

def update_reward_graph(episode_rewards):
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    plt.clf()
    plt.plot([*range(len(episode_rewards))], episode_rewards, color='red')
    plt.axhline(y = avg_reward, color = 'blue', linestyle = '--', label='avg=%s' % avg_reward)
    plt.title('Reward over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(REWARD_PATH)

def update_loss_graph(losses):
    n = len(losses['v_loss'])
    x = [*range(0, n * agent.delay_step, agent.delay_step)]
    plt.clf()
    plt.plot(x, losses['v_loss'], color='blue', label='v_loss')
    plt.plot(x, losses['q_loss'], color='red', label='q_loss')
    plt.plot(x, losses['pi_loss'], color='purple', label='pi_loss')
    plt.title('Loss over Time with Delay Step %s' % agent.delay_step)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig(LOSS_PATH)

def readCommand(argv):
    def default(s):
        return s + ' [Default: %default]'

    from optparse import OptionParser
    usageStr = """
        PURPOSE:    Begin server to transmit updated policy network weights to bot
        USAGE:      python server.py <options>
    """

    default_lr = 3e-5
    parser = OptionParser(usageStr)

    parser.add_option('-t', '--tau', dest='tau', type='float',
                      help=default('tau'), default=0.005)
    parser.add_option('-g', '--gamma', dest='gamma', type='float',
                      help=default('gamma, discount factor'), default=0.99)
    parser.add_option('-a', '--alpha', dest='alpha', type='float',
                      help=default('alpha, value learning rate'), default=default_lr)
    parser.add_option('-b', '--beta', dest='beta', type='float',
                      help=default('beta, q learning rate'), default=default_lr)
    parser.add_option('-e', '--eta', dest='eta', type='float',
                      help=default('eta, policy learning rate'), default=default_lr)
    # TODO: Make this accept a tuple
    parser.add_option('-s', '--size', dest='size', type='int',
                      help=default('image dimension'), default=256)
    parser.add_option('-B', '--batch', dest='batch_size', type='int',
                      help=default('batch size'), default=32)
    parser.add_option('--host', dest='host',
                      help=default('server hostname'), default='localhost')
    parser.add_option('--port', dest='port',type='int',
                      help=default('port number'), default=1138)
    parser.add_option('--checkpoint', dest='checkpoint',type='int',
                      help=default('checkpoint number to load'), default=-1)

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
    args['checkpoint'] = options.checkpoint

    args['host'] = options.host
    args['port'] = options.port

    return args

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from datetime import datetime
    now = str(datetime.now())
    REWARD_PATH = './logs/reward-graphs/' + now + '-reward-graph.png'
    LOSS_PATH = './logs/loss-graphs/' + now + '-loss-graph.png'
    PIC_DIR = "./logs/images/" + now + "/"
    import os
    os.mkdir(PIC_DIR)

    args = readCommand(sys.argv[1:])

    chkpt = Checkpointer(model_class=SACAgent, save_dir=models, model_params={
        "action_range"=[[-50, 50], [-60, 60]],
        "action_dim"=2,
        "gamma"=args['gamma'],
        "tau"=args['tau'],
        "v_lr"=args['alpha'],
        "q_lr"=args['beta'],
        "pi_lr"=args['eta'],
        "image_size"=(512,256,3), # args['size'],
        "kernel_size"=(3,3),
        "conv_channels"=4,
    })

    if args['checkpoint'] >= 0:
        agent = chkpt.load_checkpoint(args['checkpoint'])
    else:
        agent = SACAgent(action_range=[[-50, 50], [-60, 60]],
                        action_dim=2,
                        gamma=args['gamma'],
                        tau=args['tau'],
                        v_lr=args['alpha'],
                        q_lr=args['beta'],
                        pi_lr=args['eta'],
                        image_size=(512,256,3), # args['size'],
                        kernel_size=(3,3),
                        conv_channels=4)

    listen(agent, args['batch_size'], args['host'], args['port'])
