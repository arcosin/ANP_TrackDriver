import sys
import time
import numpy as np
from replay_buffers import BasicBuffer
sys.path.append("..")
from procnode import TCPNode

def generate_dummy_data(image_size, n=100):
    replay_buffer = BasicBuffer(int(1e6))
    for _ in range(n):
        state = np.random.randint(255, size=image_size)
        action = [np.random.random() * 100, -60 + np.random.random() * 120]
        reward = np.random.random()
        next_state = np.random.randint(255, size=image_size)
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
    
    episode_rewards = [np.random.random() * 100 for _ in range(n)]
    return replay_buffer, episode_rewards

def listen(host, port):
    client = TCPNode("client-node",False)
    client.connectTCP("server-node",host,port)
    client.start()
    while True:
        n = input("batch_size: ")
        if not n:
            print('Exiting...')
            break
        else: n = int(n)
        # Simulate sending sample of replay buffer to server
        buf, rewards = generate_dummy_data((256, 256, 3), n)
        variable = {"replay_buf": buf.buffer, "episode_rewards": rewards}
        print('variable prepared')
        start = time.time()
        client.send("server-node",variable)
        end = time.time()

        print('Data sent to server in %fs' % (end - start))

        data = client.recv()
        print('Received data from server\n')
        print(data)
    client.disconnect("server-node")
    client.stop()

def readCommand(argv):
    def default(s):
        return s + ' [Default: %default]'

    from optparse import OptionParser
    usageStr = """
        PURPOSE:    Begin robot training
        Usage:      python robot_controller.py <options>
    """
    parser = OptionParser(usageStr)
    parser.add_option('--host', dest='host', help=default('server hostname'), default='localhost')
    parser.add_option('--port', dest='port',type='int',
                      help=default('port number'), default=1138)

    options, junk = parser.parse_args(argv)
    if len(junk) != 0:
        raise Exception('Command line input not understood: ' + str(junk))

    args = dict()
    args['host'] = options.host
    args['port'] = options.port

    return args

if __name__ == "__main__":
    import sys
    args = readCommand(sys.argv[1:])
    listen(args["host"], args["port"])
