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

def test_shit():
    img = Image.open("Capture.jpg")
    img_data = np.asarray(img)

    input_shape = img_data.shape    # Should be (h, w, 3)
    num_actions = 2
    fe_filters = 4
    kernel_size = 3
    action_range = [[0, 100], [-60, 60]]
    agent = Agent(input_shape, num_actions, fe_filters, kernel_size, action_range)

    action, log_pi = agent.get_action(img_data)

    print(f"action: {action}, log_pi_type: {type(log_pi)}")


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


def robot_train(dt, agent, cam, lt, max_episodes, max_steps):
    # dt = drivetrain, cam = camera, lt = line tracker
    episode_rewards = []

    for episode in range(max_episodes):
        episode_reward = 0

        pic = cam.takePic()     #expected ndarray of (h, w, c)
        for step in range(max_steps):
            done = False
            speed, angle = agent.get_action(pic)

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
        
        # Call update after every episode, this sends over the data to PC and updates the NNs and copies then back
        # to the Rpi
        agent.update(batch_size)

    return episode_rewards