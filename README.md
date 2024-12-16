# Queens Bot

Solves the Linkedin Queens game in ~1 second using image detection and algorithms!

How it works:
 1. Open Queens on your laptop, take a screenshot of the board (no edges) and exit
 2. Move the screenshot to the boards folder and rename it to board
 3. Solve the board by running this command: <strong>python3 solve.py board</strong><br/>
    a. If you want to use ILP instead of backtracking, run: <strong>python3 solve.py board ilp</strong>
 4. View your solved board visualization
 5. Open Queens on your phone, copy the visualization and crush your friends!

<img src="example.png" style="width: 600px" />

Tools used:
 - Python (numpy, matplotlib, scipy)
