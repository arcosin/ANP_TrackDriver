import drivetrain
import camera
import linetracker

import time
import torch
import numpy as np
from PIL import Image

from models import FeatureExtractor, PolicyNetwork
from agent import Agent
from replay_buffers import BasicBuffer

import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
from procnode import TCPNode
from server import env

timestep = 1 # Seconds

def robot_train(dt, agent, cam, lt, max_episodes, max_steps, batch_size=32):
    # dt = drivetrain, cam = camera, lt = line tracker
    episode_rewards = []
    print(f"Starting training")

    replay_send_node = TCPNode("0.0.0.0", 25565)
    replay_send_node.setupServer()

    dict_recv_node = TCPNode("192.168.4.10", 25566)
    dict_recv_node.setupClient()

    for episode in range(max_episodes):
        episode_reward = 0
        print(f"Episode {episode}")

        replay_buf = BasicBuffer(int(1e6))

        pic = cam.takePic()     #expected ndarray of (h, w, c)
        for step in range(max_steps):
            done = False
            action, log_pi = agent.get_action(pic)

            speed = action[0]
            angle = action[1]

            print(f"\tUsing speed={speed}, angle={angle}")
            
            # Provide absolute speed and angle for this state, wait timestep amount of time before returning
            # NOTE: this call maintains the speed and angle after return. Subsequent calls change it. 
            dt.moveAbsoluteDelay(speed, angle, timestep)

            if lt.detect()[0] == True:
                print("\tDetected, terminate episode")
                done = True
            
            next_pic = cam.takePic()
            replay_buf.push(pic, [speed, angle], 1, next_pic, done)
            episode_reward += 1

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            pic = next_pic

        def TCP_test():
            dic = dict({"Name": "Shubham", "Friend": "Micheal"})
            print("Sending replay buffer (temp dict)...")
            replay_send_node.send(dic)
            print("Sent!")
            time.sleep(5)
            print("Attempting to receive dictionary...")
            wowDict = dict_recv_node.recv()
            print(f"Received dictionary: {wowDict}")

        def pickle_test():
            print("Generating sample from replay buffer of size %s..." % batch_size)
            start = time.time()
            replay_buf_sample = replay_buf.sample(batch_size)
            end = time.time()
            print("Generated sample in %fs\n" % (end - start))

            print("Pickling sample...")
            start = time.time()
            replay_buf_sample = pickle.dumps(replay_buf_sample)
            end = time.time()
            print("Pickled sample in %fs\n" % (end - start))

            print("Connecting to server...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((env.HOST, env.PORT))
            print("Connected")

            print("Sending sample to server...")
            start = time.time()
            s.sendall(replay_buf_sample)
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
            print("Received updated models from server in %fs" % (end - start))

            print("Closing connection...\n")
            s.close()

        pickle_test()

    return episode_rewards

if __name__ == "__main__":
    print("Initializing objects")
    dt = drivetrain.DriveTrain()
    cam = camera.Camera()
    lt = linetracker.LineTracker()

    input_shape = (256, 256, 3)    # Should be (h, w, c)
    num_actions = 2
    fe_filters = 4
    kernel_size = 3
    action_range = [[0, 100], [-60, 60]]

    agent = Agent(input_shape, num_actions, fe_filters, kernel_size, action_range)
    robot_train(dt, agent, cam, lt, 1, 3)

