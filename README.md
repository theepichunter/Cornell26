# Pool Project

Authors: Jeremy Shen, Hunter Pesin

Description: Python program that emulates a rudimentary version of 8ball using the physics of simple collisions. This program uses Pymunk and Pygame solely for the purpose of simulating the balls. All position updating and collision handling are calculated manually. We attempt to best replicate the rules or 8ball, but we are by no means expert 8ball players.

Installation Requirements: Unix; download and unzip the executables, either click on the executable from the file explorer or use command line `./` Depending on your device's setup (specifically anti-malware software), it might be necessary to grant executable permissions `chmod +x pool`

Usage: Before starting the users are prompted to set the friction coefficient and elasticity constant.* To hit the cue ball while playing, the player clicks in the direction that they wish to hit the ball. The distance between the click and the cue ball determines the magnitude of the impulse applied and consequently initial velocity of the cue ball (up to a preset maximum).

Refer to attached video for a video demonstration of gameplay

Click on the game screen and then press Q twice while the balls are moving to quit the game. While doing so, plots of total kinetic energy and trajectory of the center of mass throughout the game will appear. In the center of mass trajectory plot, darker points indicate the location of the COM at earlier timesteps within one player's move, and points get lighter as time progresses in one player's move.

*Make sure to drag in order to set the sliders. 

Limitations: The simulation doesn't take into account rolling vs sliding; it’s 
assumed that all balls are sliding across the table. The runtime speed is below average also 
fairly slow, and each turn takes quite long to end.
