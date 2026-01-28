## Initial Human Prompt
I want you to read the deepmind open_spiel's alpha-zero implementation and create an alpha-zero chess engine from scratch. Read the API documentation and structure an minimal chess engine without calling the open_spiel's library. The purpose of the project is as follows:
create an efficient yet still understandable project to help understand and practice the theory from alpha-zero paper
Do not use the open_spiel library. Use it only as a structural reference for best practices (e.g., how they handle MCTS and state transitions). You can refer part of the code if needed.
Prioritize an elegant, readable codebase that maps directly to the AlphaZero paper's mathematical concepts, but optimize it for a single-node setup (High-end CPU + NVIDIA GPU).
The architecture must support the depth and scaling required to eventually reach Grandmaster-level play (Elo 2500+).
The training process should be as efficient as possible (use cpp + multi processing if needed) and the framework should be elegant and efficient.
The structure of the project should be as elegant as possible.

## Second Edition
Role: You are an expert AI Research Engineer specializing in Reinforcement Learning and Computer Games.

Task: Design a "from-scratch" AlphaZero implementation specifically for Chess, using Pytorch, a chess library of your choice (python-chess or other) and some cpp (implement in cpp only if the performance of the component is critical for scaling, otherwise use python)

I want you to read the deepmind open_spiel's alpha-zero implementation and create an alpha-zero chess engine from scratch. Read the API documentation and structure an minimal chess engine without calling the open_spiel's library. The purpose of the project is as follows:
Create an efficient yet still understandable project to help understand and practice the theory from alpha-zero paper
2. Do not use the open_spiel library. Use it only as a structural reference for best practices (e.g., how they handle MCTS and state transitions). You can refer part of the code if needed.
3. Prioritize an elegant, readable codebase that maps directly to the AlphaZero paper's mathematical concepts, but optimize it for a single-node setup (High-end CPU + NVIDIA GPU).
4. The architecture must support the depth and scaling required to eventually reach Grandmaster-level play (Elo 2500+).
5. The training process should be as efficient as possible (use cpp + multi processing if needed) and the framework should be elegant and efficient.
6. The structure of the project should be as elegant as possible.