## Initial Human Prompt (~$5)
I want you to read the deepmind open_spiel's alpha-zero implementation and create an alpha-zero chess engine from scratch. Read the API documentation and structure an minimal chess engine without calling the open_spiel's library. The purpose of the project is as follows:
create an efficient yet still understandable project to help understand and practice the theory from alpha-zero paper
Do not use the open_spiel library. Use it only as a structural reference for best practices (e.g., how they handle MCTS and state transitions). You can refer part of the code if needed.
Prioritize an elegant, readable codebase that maps directly to the AlphaZero paper's mathematical concepts, but optimize it for a single-node setup (High-end CPU + NVIDIA GPU).
The architecture must support the depth and scaling required to eventually reach Grandmaster-level play (Elo 2500+).
The training process should be as efficient as possible (use cpp + multi processing if needed) and the framework should be elegant and efficient.
The structure of the project should be as elegant as possible.

## Second Edition (~$25)
Role: You are an expert AI Research Engineer specializing in Reinforcement Learning and Computer Games.

Task: Design a "from-scratch" AlphaZero implementation specifically for Chess, using Pytorch, a chess library of your choice (python-chess or other) and some cpp (implement in cpp only if the performance of the component is critical for scaling, otherwise use python)

I want you to read the deepmind open_spiel's alpha-zero implementation and create an alpha-zero chess engine from scratch. Read the API documentation and structure an minimal chess engine without calling the open_spiel's library. The purpose of the project is as follows:
Create an efficient yet still understandable project to help understand and practice the theory from alpha-zero paper
2. Do not use the open_spiel library. Use it only as a structural reference for best practices (e.g., how they handle MCTS and state transitions). You can refer part of the code if needed.
3. Prioritize an elegant, readable codebase that maps directly to the AlphaZero paper's mathematical concepts, but optimize it for a single-node setup (High-end CPU + NVIDIA GPU).
4. The architecture must support the depth and scaling required to eventually reach Grandmaster-level play (Elo 2500+).
5. The training process should be as efficient as possible (use cpp + multi processing if needed) and the framework should be elegant and efficient.
6. The structure of the project should be as elegant as possible.

## [Generate Project Plan](project_plan.md)

## Second generation prompt (~$10)

I have identify the following error from the code
    - The training script cannot stop when there's error in the terminal nor can the script be stopped by control+C
    - Log the error during training in training,log, currently the error that happens when running the code is not
    - The root cause is a bug in the code that handles pawn underpromotions (promoting to a piece other than a queen). The logic for decoding these special moves is flawed and doesn't correctly handle captures. When the MCTS selects a legal underpromotion that is also a capture, the decoder misinterprets it as a simple (and illegal) forward pawn move, causing the crash.
    - an inconsistency in how "near-greedy" temperatures are handled. A temperature of 0.01 is returned for exploitation, but the code that checks for it uses < 0.01, which excludes 0.01 itself. This causes the calculation to proceed with power(counts, 100), leading to an overflow.
    - Add Cython and C++ backend for MCTS

## Third generation prompt (~$10)
Continue with the project plan you proposed:
    - Batched GPU inference (amortizes Python overhead)
    - Virtual loss for parallel MCTS (tree operations become bottleneck)

## [Generate Google Colab Project Plan](/Google%20Colab/project_plan.md)

## Fourth

Training loop add iterations.

create and maintian a document bug_fix.md and review the preivous mistake before you start writing new code

Open books in epd: https://github.com/fairy-stockfish/books/tree/master

