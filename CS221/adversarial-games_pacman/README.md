# Pacman

<div>

Pac-Man is a game where Pac-Man moves around in a maze and tries to eat as many _food pellets_ (the small white dots) as possible, while avoiding the ghosts (the other two agents with eyes in the above  gure). If Pac-Man eats all the food in a maze, it wins. The big white dots at the top-left and bottom-right corner are _capsules_, which give Pac-Man power to eat ghosts in a limited time window.

In this project, we design agents for the classic version of Pac-Man, including ghosts. Along the way, we implement both minimax and expectimax search.

## Code and Running the Tester

All code is entered into `src/runmodel.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

`$ python tester.py`