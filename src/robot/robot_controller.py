import drivetrain
import camera
import linetracker

import time
import torch
import numpy as np
from PIL import Image

from models import FeatureExtractor, PolicyNetwork
from agent import Agent

timestep = 1 # Seconds

def robot_train(dt, agent, cam, lt, max_episodes, max_steps):
    # dt = drivetrain, cam = camera, lt = line tracker
    episode_rewards = []
    print(f"Starting training")

    for episode in range(max_episodes):
        episode_reward = 0
        print(f"Episode {episode}")

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
            #agent.replay_buffer.push(pic, [speed, angle], 1, next_pic, done)
            episode_reward += 1

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            pic = next_pic
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
    robot_train(dt, agent, cam, lt, 10, 5)

