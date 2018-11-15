# jumble-puzzle
A jumble puzzle solver using apacke spark and python
The jumble puzzle is a common newspaper puzzle, it contains a series of anagrams that must be solved (see https://en.wikipedia.org/wiki/Jumble).

## Solution(s)
 ### solution_greedy.py
 This solution follows the greedy approach of returning the list of words which have lowest frequency value(individually) for that size in the frequency dictionary.
 e.g. for image 1, the approach returns:['and', 'more', 'call'] which have the lowest frequency value individually.
 
 ### solution_better.py
 This solution is more complete solution than the greedy solution. While greedy solution is faster to run, it ignores a plethora of combinations of words. This solution takes into consideration all possible combinations of segment words possible with each other, and provides five best solutions. This solution is also flexible in stopping condition via the use of parameter `SCORE_THRESHOLD`, which can be learnt and adjusted iteratively.
 
 ## Running the solutions
 ### Pre-requisites
 1. pyspark 2.xx (2.4 used while writing above code)
 2. Python 3.x.x (3.6.5 used while writing above code
 
 ### Run solutions
 `python solution_greedy.py`
 
 `python solution_better.py`
