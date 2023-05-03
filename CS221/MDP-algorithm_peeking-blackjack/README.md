# Peeking Blackjack 📝

<div>

 After exploring search algorithms in the [previous project](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/search-algorithm_text-reconstruction), we now delve into an environment where the outcomes of actions are uncertain. This is a common scenario in the real world, making the ability to reason amidst uncertainty a key aspect of an effective AI system.

Markov decision processes (MDPs) can be used to formalize uncertain situations. Here, we implement algorithms to find the optimal policy under these conditions. Our subject of study is a modified version of Blackjack, which we formalize as an MDP and apply our algorithm to discover the optimal policy.

### Game Details 🃏

For this problem, we create an MDP to describe states, actions, and rewards in this game.

In this Blackjack variation, the deck can contain an arbitrary collection of cards with distinct face values. At the beginning of the game, the deck contains an equal number of cards for each face value, termed as the _'multiplicity'_. For example, a standard 52-card deck would have face values ranging from `[1, 2, ..., 13]` and multiplicity 4. On the other hand, a deck with face values of `[1, 5, 20]` with a multiplicity of 10 would contain 30 cards in total (10 each of 1s, 5s, and 20s). The deck is shuffled, meaning that each permutation of the cards is equally probable.

![image](https://user-images.githubusercontent.com/45148177/235787182-fcf971b2-d558-43b4-b2f3-681492d27507.png)

The game proceeds in a sequence of rounds. In each round, the player can _(i)_ draw the next card from the top of the deck (cost-free), _(ii)_ peek at the top card (costing `peekCost`, with the same card to be drawn in the next round), or _(iii)_ quit the game. (Note: it isn't possible to peek twice in a row; if attempted `succAndProbReward()` should return [].)

The game continues until one of the following conditions becomes true:

- The player quits, earning a reward equal to the sum of the face values of the cards in their hand;
- The player draws a card and _"goes bust"_, i.e., the sum of the face values of the cards in their hand exceeds the threshold set at the game's start. This results in a reward of 0;
- The deck runs out of cards, treated as if the player quit and they earn a reward equal to the sum of the cards in their hand. _Note: if you take the last card and go bust, then the reward becomes 0 not the sum of card values.

## Code and Running the Tester

All code is entered into `src/model.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

```bash
$ python tester.py
```
