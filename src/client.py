# NOTE: This is purely a test file that is used to demonstrate pickle functionality.
#       It mimics the pipeline that `robot_controller` will follow, but artificially
#       generates a replay_buffer instead of physically generating data to send.
#       This file should be deleted once we decide how we want to implement IPC.

import socket, pickle, time
import numpy as np
from server import env
from sac import BasicBuffer

def generate_dummy_buffer(image_size=(256, 256, 3), n=100):
    replay_buffer = BasicBuffer(int(1e6))
    for _ in range(n):
        state = np.random.randint(255, size=image_size)
        action = [np.random.random() * 100, -60 + np.random.random() * 120]
        reward = np.random.random()
        next_state = np.random.randint(255, size=image_size)
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
    return replay_buffer

def pickle_test(batch_size=2):
    replay_buf = generate_dummy_buffer(n=5)

    print("Pickling buffer...")
    start = time.time()
    replay_buf = pickle.dumps(replay_buf.buffer)
    end = time.time()
    print("Pickled buffer in %fs\n" % (end - start))

    print("Connecting to server...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((env.HOST, env.PORT))
    print("Connected")

    print("Sending sample to server...")
    start = time.time()
    s.sendall(replay_buf)
    end = time.time()
    print("Sample sent to server in %fs" % (end - start))

    print("Closing connection...\n")
    s.close()

    # NOTE: I was not able to get this to work with the same connection,
    #       so I close it and make a new one. It might be possible to use the same connection.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((env.HOST, env.PORT))
    data = []
    print("Receiving updated models from server...")
    start = time.time()
    while True:
        packet = s.recv(1024)
        if not packet: break
        data.append(packet)
    updated_fe, updated_pi = pickle.loads(b"".join(data))
    end = time.time()
    print(updated_fe)
    print('\n\n\n\n')
    print(updated_pi)
    print("Received updated models from server in %fs" % (end - start))

    print("Closing connection...\n")
    s.close()

pickle_test()
