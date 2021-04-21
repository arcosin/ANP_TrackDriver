import socket, pickle, time
import env
import numpy as np
from replay_buffers import BasicBuffer

def generate_dummy_buffer(image_size, n=100):
    replay_buffer = BasicBuffer(int(1e6))
    for _ in range(n):
        state = np.random.randint(255, size=image_size)
        action = [np.random.random() * 100, -60 + np.random.random() * 120]
        reward = np.random.random()
        next_state = np.random.randint(255, size=image_size)
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
    print('Finished generating buffer')
    return replay_buffer

while True:
    n = input("batch_size: ")
    if not n:
        print('Exiting...')
        break
    else: n = int(n)
    # Simulate sending sample of replay buffer to server
    variable = generate_dummy_buffer((256, 256, 3)).sample(n)
    data_string = pickle.dumps(variable)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((env.HOST, env.PORT))

    start = time.time()
    s.sendall(data_string)
    end = time.time()

    print('Data sent to server in %fs' % (end - start))
    s.close()

    # Simulate receiving updated models from server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((env.HOST, env.PORT))
    data = []
    while True:
        packet = s.recv(1024)
        if not packet: break
        data.append(packet)
    data = pickle.loads(b"".join(data))
    print('Received data from server\n')

    s.close()

