# Scheduling 📝

<div>

Deciding which courses to take each quarter can be a complex task. It's a balancing act that involves considering your interests, graduation requirements, prerequisite chains, and the availability of each course. To simplify this process, we've developed an automated course scheduling program that takes into consideration all these factors.

In this project, we cast the course scheduling problem as a constraint satisfaction problem (CSP). The program considers various constraints such as course prerequisites, time slots, and the student's preferences to generate an optimal course schedule. It utilizes backtracking search to solve this CSP, providing students with the most suitable course schedule for their needs.

## Code and Running the Tester 🛠️

All code is entered into `src/model.py`. The unit tests in `src/tester.py` (the autograder) is used to verify a correct script. Run the tester locally using the following terminal command within the `src/` subdirectory:

```bash
$ python tester.py
```
