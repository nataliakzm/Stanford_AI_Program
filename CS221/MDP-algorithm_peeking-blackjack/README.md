# Peeking Blackjack

<div>

The search algorithms explored in the (previous project)[] work great when we know exactly the results of the actions. Unfortunately, the real world is not so predictable. One of the key aspects of an effective AI is the ability to reason in the face of uncertainty.

Markov decision processes (MDPs) can be used to formalize uncertain situations. In this project, we implement algorithms to find the optimal policy in these situations. We then formalize a modified version of Blackjack as an MDP, and apply the algorithm to find the optimal policy.

### Game Details

For this problem, we create an MDP to describe states, actions, and rewards in this game.

For this version of Blackjack, the deck can contain an arbitrary collection of cards with different face values. At the start of the game, the deck contains the same number of each cards of each face value; we call this number the _'multiplicity'_. For example, a standard deck of 52 cards would have face values `[1, 2, ..., 13]` and multiplicity 4. We
could also have a deck with face values `[1, 5, 20]`; if we used multiplicity 10 in this case, there would be 30 cards in total (10 each of 1s, 5s, and 20s). The deck is shuffled, meaning that each permutation of the cards is equally likely.

![image](https://user-images.githubusercontent.com/45148177/235787182-fcf971b2-d558-43b4-b2f3-681492d27507.png)

The game occurs in a sequence of rounds. Each round, the player either _(i)_ takes the next card from the top of the deck (costing nothing), _(ii)_ peeks at the top card (costing `peekCost`, in which case the next round, that card will be drawn), or _(iii)_ quits the game. (Note: it is not possible to peek twice in a row; if the player peeks twice in a row, then `succAndProbReward()` should return [].)

The game continues until one of the following conditions becomes true:

- The player quits, in which case her reward is the sum of the face values of the cards in her hand.
- The player takes a card and "goes bust". This means that the sum of the face values of the cards in her hand is strictly greater than the threshold specified at the start of the game. If this happens, her reward is 0.
- The deck runs out of cards, in which case it is as if she quits, and she gets a reward which is the sum of the cards in her hand. _Make sure that if you take the last card and go bust, then the reward becomes 0 not the sum of values of cards_.

## Code and Running the Tester

All code is entered into `src/model.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

`$ python tester.py`
