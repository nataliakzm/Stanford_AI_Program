# CS221: Artificial Intelligence Principles and Techniques 🚀

<div>

This repository contains ptojects created as part of the Stanford's CS221: AI Principles and Techniques course. 

During the course, I developed a strong grasp of foundational AI principles and techniques, which included:

- Understanding foundational AI principles such as ML, state-based models, variable-based models, and logic. I also implemented search algorithms for finding shortest paths, planning robot motions, and performing machine translation;
- Finding optimal policies in uncertain situations using Markov decision processes (MDPs);
- Designing agents and optimizing strategies in adversarial games, such as Pac-Man;
- Adapting to preferences and limitations using constraint satisfaction problems (CSPs);
- Predicting likelihoods of causes with Bayesian networks;
- Defining logic in algorithms with syntax, semantics, and inference rules;


## Getting Started 🧭

### Installation

To get started with the course materials, clone this repository to your local machine:

```bash
$ git clone https://github.com/nataliakzm/Stanford_AI_Program.git
```

Next, navigate into the cloned repository and create a new conda environment from the `environment.yml` file, by running:

```bash
cd Stanford_AI_Program/CS221/<project>
`$ conda env create -f src/environment.yml
```

Once you've set up your environment, activate it:

```bash
`$ conda activate [your-environment-name]
```

### Usage

Each project from the course is located in its own directory within the repository. To get started with a project, navigate into the corresponding directory and follow the instructions in the README.

- **Project 1** – [**Sentiment Analysis**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/ML_sentiment_analysis): Built a binary linear classifier to analyze movie reviews and determine whether they're _"positive"_ or _"negative"_. Also created a K-means clustering model.
- **Project 2** – [**Text Reconstruction**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/search-algorithm_text-reconstruction): Worked on word segmentation and vowel insertion, which is relevant in languages like Chinese or Arabic, where words might not be flanked by spaces, or vowels are often not written.  
- **Project 3** – [**Peeking Blackjack**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/MDP-algorithm_peeking-blackjack): Used Markov decision processes (MDPs) to find the optimal policy in a modified version of Blackjack.
- **Project 4** – [**Multi-agent Pac-Man**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/adversarial-games_pacman): Designed agents for the classic version of Pac-Man, implementing both minimax and expectimax search.
- **Project 5** – [**Scheduling**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/CSP_course-scheduling): reated a program that automatically schedules courses based on preferences and constraints, using backtracking search to solve the CSP for an optimal course schedule.
- **Project 6** – [**Car Tracking**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/bayesian-networks_car-tracking): Focused on the sensing system of an autonomous driving system, tracking other cars based on noisy sensor readings.
- **Project 7** – [**Logic**](https://github.com/nataliakzm/Stanford_AI_Program/tree/main/CS221/DL_logic): Gained hands-on experience with logic, using it to represent the meaning of natural language sentences, solve puzzles, and prove theorems. Most of this project involved translating English into logical formulas, but also included exploring the mechanics of logical inference.
