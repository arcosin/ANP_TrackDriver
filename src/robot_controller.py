import sys
import time
import torch
import pickle
import socket
import random
import threading
import numpy as np
from PIL import Image

from robot import Agent
from robot import Camera
from robot import DriveTrain
from robot import LineTracker

# from os import path

# sys.path.append(path.join(path.dirname(__file__), '..'))
from sac import BasicBuffer

timestep = 0.125 # Seconds

class thread(threading.Thread):
    def __init__(self, name, id, lt):
        threading.Thread.__init__(self)
        self.name = name
        self.id = id
        self.lt = lt
        self.detected = False
        self.kill = False

    def run(self):
        while not self.kill:
            if lt.detect()[0] == True:
                self.detected = True
                time.sleep(1)

def pickle_test(replay_buf, episode_rewards, host, port):
    data = {"replay_buf": replay_buf.buffer, "episode_reward": episode_rewards[-1]}
    print("Pickling buffer and episode rewards...")
    start = time.time()
    data = pickle.dumps(data)
    end = time.time()
    print("Pickled buffer and episode rewards in %fs\n" % (end - start))

    print("Connecting to server...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print("Connected")

    print("Sending sample to server...")
    start = time.time()
    s.sendall(data)
    end = time.time()
    print("Sample sent to server in %fs" % (end - start))

    print("Closing connection...\n")
    s.close()

    # NOTE: I was not able to get this to work with the same connection,
    #       so I close it and make a new one. It might be possible to use the same connection.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
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
    agent.fe.load_state_dict(updated_fe)
    agent.pi.load_state_dict(updated_pi)
    print("Updated agent models")
    print("Closing connection...\n")
    s.close()

def robot_train(dt, agent, cam, lt, max_episodes, max_steps, batch_size, host, port):
    episode_rewards = []
    print(f"Starting training")

    action_stack =[]
    for episode in range(max_episodes):
        episode_reward = 0
        print(f"Episode {episode}")

        replay_buf = BasicBuffer(int(1e6))

        pic = cam.takePic()     #expected ndarray of (w, h, c)
        action_stack.clear()

        detector = thread("detector", 1000, lt)
        detector.start()

        speed = 0
        angle = 0
        first_step = True

        sent = False
        for step in range(max_steps):
            done = False

            rescaled_action, action = agent.get_action(pic)

            old_speed = speed
            old_angle = angle

            speed = rescaled_action[0]
            angle = rescaled_action[1]


            # Provide absolute speed and angle for this state, wait timestep amount of time before returning
            # NOTE: this call maintains the speed and angle after return. Subsequent calls change it.

            step_end = time.time()
            if first_step == True:
                first_step = False
            else:
                action_stack.append((old_speed, old_angle, step_end - step_start))
                print(f"\t\tAction took {step_end - step_start} seconds")

            print(f"\tUsing speed={speed}, angle={angle}")
            step_start = time.time()
            dt.moveAbsoluteDelay(speed, angle, timestep)

            if detector.detected == True:
                print("\tDetected, terminate episode")
                done = True
                episode_reward += -1000   #Adjust
            else:
                episode_reward += speed
            next_pic = cam.takePic()
            replay_buf.push(pic, action, 1, next_pic, done)

            if step == max_steps - 1:
                dt.driveHalt()

                step_end = time.time()
                action_stack.append((speed, angle, step_end - step_start))

                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break
            elif done:
                # TODO: Fix rollback procedure
                #print("\tStarting automatic rollback")
                dt.driveHalt()
                step_end = time.time()
                action_stack.append((speed, angle, step_end - step_start))

                print(f"\tLast action took {step_end - step_start} seconds")

                input("Press enter to rollback")
                robot_rollback(action_stack)
                #if lt.detect()[0]:
                #    print("\tRollback failed! Please reset the bot back to the track manually")

                dt.driveHalt()

                #input("HERE TO BREAK")

                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))

                sent = True
                detector.kill = True
                pickle_test(replay_buf, episode_rewards, host, port)
                input("Press enter once the robot is reset on the track")
                break

            pic = next_pic

        if not sent:
            detector.kill = True

            pickle_test(replay_buf, episode_rewards, host, port)

    return episode_rewards

def robot_rollback(action_stack):
    # rollback random number of steps to set the robot back to track
    #t = min(random.randint(0, len(action_stack) - 1), 5)
    t = len(action_stack)
    for i in range(t):
        speed, angle, timestep = action_stack.pop()
        dt.moveAbsoluteDelay(-speed, angle, timestep)

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
    args = readCommand(sys.argv[1:])
    print("Initializing objects")
    dt = DriveTrain()
    cam = Camera()
    lt = LineTracker()

    input_shape = (512, 256, 3)    # Should be (h, w, c)
    num_actions = 2
    fe_filters = 4
    kernel_size = 3
    action_range = [[-30, 50], [-60, 60]]

    agent = Agent(input_shape, num_actions, fe_filters, kernel_size, action_range)
    robot_train(dt, agent, cam, lt,
                max_episodes=50, max_steps=50, batch_size=32,
                host=args['host'], port=args['port'])
