--------------------------------------------------------------------------------
# ANP_TrackDriver
--------------------------------------------------------------------------------
Repository for the advanced neural projects group's track driving robot project.

**Robot:**
A rover style robot with a camera and line reader on the front. Optimally, the camera should be mounted at a 60 degree angle from the ground. The line reader should be nearly flush to the ground so it can detect the line.

**Environment:**
The environment is a flat open space with a closed-loop track formed out of black tape. It should be reasonably well lit. An outer and inner line of bounding tape should concentrically define the track with an offset of 2 times the width of the robot’s chassis.

**Task:**
The robot should navigate the track continuously, strictly staying within the bounding lines until the end of the episode. When the episode ends, the robot should be replaced in the track at any position facing forward with its line sensor close to the center of the track. The robot should only use the camera to navigate the track. The line sensor should only be used to detect a failure and should not be the input to any neural nets.

**Reward:**
- The default reward at each timestep is 0.
- The agent receives +1 reward every time step that it has moved forward with both motors (this is not triggered while in a deadzone around 0 speed or going backwards)
- The agent loses a large amount of reward (e.g. -1000) whenever its line sensor detects a bounding line. The robot also stops and the episode ends. This simulates a wall collision.
- If the robot has not exceeded the bounding tape in a set number of steps (e.g. 10000) the episode ends without a reward loss.


**Supervisors:**
- Maxwell Joseph Jacobson
- Dr. Gustavo Rodriguez-Rivera


**Contributors:**
- \[add your name here if you are a contributor\]


--------------------------------------------------------------------------------
