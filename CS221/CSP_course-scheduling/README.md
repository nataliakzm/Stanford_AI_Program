# Course Scheduling

<div>

What courses should you take in a given quarter? Answering this question requires balancing your interests, satisfying prerequisite chains, graduation requirements, availability of courses; this can be a complex tedious process.

In this project, we write a program that does automatic course scheduling for a student based on the preferences and constraints. The program will cast the course scheduling problem (CSP) as a constraint satisfaction problem (CSP) and then use backtracking search to solve that CSP to give a student an optimal course schedule.

## Code and Running the Tester

All code is entered into `src/model.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

`$ python tester.py`