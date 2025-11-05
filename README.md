SECTION A – Theory Questions (Short Answers)

Q1. Explain polynomial time reducibility with example.
Ans1 - Polynomial time reducibility ($A \le_p B$) means an instance of problem $A$ can be transformed into an instance of problem $B$ in polynomial time such that $A$ is 'yes' if and only if $B$'s transformed instance is 'yes'.Example: CLIQUE $\le_p$ INDEPENDENT SET. A graph $G$ has a $k$-clique if and only if its complement graph $\bar{G}$ has a $k$-independent set.

Q2. Define and differentiate: NP Packet Graph & NP Scheduling Problem
   Feature                          NP Packet Graph (e.g., edge-disjoint path)          NP Scheduling Problem (e.g., job shop)
   
   DefinitionConcerns               Routing/flow of data packets in a network;          Concerns optimal allocation of tasks (jobs)
                                    typically asks if communication demands can         to resources (machines) over time to minimize cost or completion time.
                                    be met under capacity/time constraints.
   
   Core Idea                        Connectivity and resource contention in a           Optimization and constraint satisfaction over a timeline.
                                    network graph.

Q3. What is a non-deterministic algorithm? Give one example.
Ans3 - A non-deterministic algorithm is a theoretical model that can "guess" the correct next step among multiple possibilities and always chooses the path leading to a solution, if one exists, in polynomial time.
       Example: An algorithm for the Satisfiability (SAT) Problem that non-deterministically guesses a truth assignment for all variables, and then deterministically checks the assignment.

Q4. What is a state space tree? State two ways to search for an answer node.
Ans4 - A state space tree is a tree representation of all possible sequences of decisions (or states) an algorithm can take from an initial state to a goal state to solve a problem.
       Two ways to search: Breadth-First Search (BFS) and Depth-First Search (DFS) (used in Backtracking/Branch & Bound).

Q5. Explain Boolean Satisfiability (SAT) problem with one example.
Ans5 - SAT is the decision problem of determining if there exists an assignment of truth values (TRUE/FALSE) to the variables of a given Boolean formula (usually in CNF) that makes the entire formula evaluate to           TRUE.Example: Is $$(x_1 \lor \neg x_2) \land (\neg x_1 \lor x_2)$$ satisfiable? Yes, $x_1=\text{TRUE}, x_2=\text{TRUE}$ works.

Q6. Define Cook’s Theorem in one line and state one application.
Ans6 - Cook's Theorem states that the Boolean Satisfiability Problem (SAT) is NP-complete.
       Application: It is the foundational problem used to prove that countless other problems are NP-complete via polynomial-time reduction.

Q7. Differentiate: P, NP, NP-complete, NP-hard & Write two examples for each class.
Ans7 - Class	                              Definition (Brevity)	                                    Two Examples (short titles only)
         P	               Solvable by a deterministic algorithm in polynomial time.	     Minimum Spanning Tree, Single-Source Shortest Path
        NP	      Solutions can be verified by a deterministic algorithm in polynomial     SAT, Traveling Salesperson Decision
                  time.
    NP-Complete	  Problems in NP to which every problem in NP can be reduced.	             Boolean Satisfiability (SAT), 3-Coloring
     NP-Hard	    Problems to which every problem in NP can be reduced.	                   Traveling Salesperson Optimization, Halting Problem

Q8. Explain the Hamiltonian Cycle problem in context of NP.
Ans8 - The Hamiltonian Cycle (HC) problem asks if a graph contains a cycle that visits every vertex exactly once. It is a classic NP-complete problem. It is in NP because a proposed cycle can be verified quickly         (polynomial time), and it is NP-hard because problems like SAT can be reduced to it.

SECTION B – Algorithm Applications & Coding

Q9. Dijkstra’s Algorithm
- State the core logic in one line.
- Write pseudocode or main function steps only.
Ans9 - Core Logic: It finds the shortest path from a single source node to all other nodes in a graph with non-negative edge weights by greedily selecting the unvisited vertex with the smallest tentative distance.
Pseudocode : 
       function DIJKSTRA(Graph G, Source s):
    // 1. Initialization
    dist[s] = 0
    for all v in G, v != s: dist[v] = infinity
    Q = set of all vertices in G

    // 2. Main Loop
    while Q is not empty:
        u = vertex in Q with min dist[u]
        Remove u from Q

        // 3. Relaxation
        for each neighbor v of u:
            if dist[u] + weight(u, v) < dist[v]:
                dist[v] = dist[u] + weight(u, v)  

Q10. N-Queen Problem
- Write only the key recursive function.
- Add minimal comments for clarity.
Ans10 - Key Recursive Function :
def solveNQueens(n):
    board = [[0]*n for _ in range(n)]

    def solve_util(row):
        # Base case: All queens are placed (last row has been placed)
        if row >= n:
            return True 

        # Try placing queen in each column of the current row
        for col in range(n):
            if is_safe(board, row, col, n):
                board[row][col] = 1 # Place Queen (Tentative decision)

                # Recurse for the next row
                if solve_util(row + 1):
                    return True

                # Backtrack: If placing queen at (row, col) doesn't lead to a solution
                board[row][col] = 0 # Remove Queen and try next column

        return False # No safe column found in this row

    # is_safe function (not fully shown) checks column, 45-diag, 135-diag
    # call: solve_util(0)

Q11. Dynamic Programming – 0/1 Knapsack Problem
- Weights: {3, 4, 6, 5}
- Profits: {2, 3, 1, 4}
- Capacity: 8
- Write a DP function/pseudocode and print total profit.
Ans11 - Problem Data: Weights: {3, 4, 6, 5}, Profits: {2, 3, 1, 4}, Capacity: 8
        DP Function/Pseudocode and Total Profit Calculation:$$W = \{3, 4, 6, 5\}, P = \{2, 3, 1, 4\}, C = 8, n=4
        Let $DP[i][w]$ be the maximum profit with the first $i$ items and capacity $w$.$$DP[i][w] = \begin{cases} DP[i-1][w] & \text{if } W_i > w \\ \max(DP[i-1][w], P_i + DP[i-1][w - W_i])
        {if } W_i \le          w \end{cases}$$
  
  def knapsack_dp(W, P, C):
    n = len(W)
    # Initialize DP table (n+1 rows for items, C+1 columns for capacity)
    DP = [[0] * (C + 1) for _ in range(n + 1)]

    # Iterate through items (i) and capacities (w)
    for i in range(1, n + 1):
        weight_i = W[i-1] # Current item weight
        profit_i = P[i-1] # Current item profit

        for w in range(C + 1):
            # Case 1: Item 'i' cannot be included (weight too high)
            if weight_i > w:
                DP[i][w] = DP[i-1][w]
            # Case 2: Item 'i' can be included (take max of excluding vs. including)
            else:
                # Excluding: Profit from previous item at capacity w
                profit_exclude = DP[i-1][w] 
                # Including: Profit of current item + max profit from remaining capacity
                profit_include = profit_i + DP[i-1][w - weight_i] 
                DP[i][w] = max(profit_exclude, profit_include)

    total_profit = DP[n][C]
    print(f"Total Profit: {total_profit}")
    # Return the bottom-right cell value
    return total_profit

# Call the function with the given data
weights = [3, 4, 6, 5]
profits = [2, 3, 1, 4]
capacity = 8
# knapsack_dp(weights, profits, capacity)
# The calculated value is 7 (Items 1(W=3,P=2) and 4(W=5,P=4) -> W=8, P=6;
# OR Items 2(W=4,P=3) and 4(W=5,P=4) -> W=9, too high;
# OR Items 1(W=3,P=2) and 2(W=4,P=3) -> W=7, P=5;
# Best is Items 2 and 4 (Wait, 4+5=9 > 8).
# Correct Best: Item 1 (W=3, P=2) + Item 2 (W=4, P=3) + Item 3 (W=6, P=1) [No, 3+4+6=13]
# Re-eval: Max is P=7 (Items 4 and 1: 5+3=8, 4+2=6).
# P=3+4=7 (Items 2 and 4: 4+5=9 > 8).
# P=2+3=5 (Items 1 and 2: 3+4=7 < 8).
# P=4+2=6 (Items 4 and 1: 5+3=8).
# P=4+3=7 (Items 4 and 2: 5+4=9 > 8).
# The max profit is **6** (using items with W=5, P=4 and W=3, P=2).

# Final Answer: Total Profit: 6

Q12. Travelling Salesman Problem – Branch & Bound
- Take nodes A, B, C, D with edge weights (example: AB=10, AC=6, AD=8, BC=5, BD=9, CD=7).
- State essential pseudocode steps to solve this problem.
Ans12 - Essential Pseudocode Steps :
        function TSP_BRANCH_AND_BOUND(Graph G):
    # 1. Initialization
    Min_Cost = infinity # Global minimum path cost found so far
    PriorityQueue Q = new PriorityQueue() # Min-heap of partial tours

    # 2. Initial State & Lower Bound
    # Calculate initial Lower Bound (LB) of the whole graph (e.g., Row Reduction)
    Root_Node = (path={}, cost=0, lower_bound=Initial_LB)
    Q.insert(Root_Node)

    # 3. Search Loop
    while Q is not empty:
        u = Q.extract_min() # Extract node with smallest LB

        # Pruning Step: If current LB already exceeds global Min_Cost, skip
        if u.lower_bound >= Min_Cost:
            continue 

        # 4. Expansion (Branching)
        for each unvisited neighbor v of u:
            v_path = u.path + {v}
            v_cost = u.cost + weight(u, v)

            # Calculate new Lower Bound (LB) for the child node v
            v_lower_bound = calculate_LB(v_path, v_cost) 

            # 5. Pruning and Recording
            if v_lower_bound < Min_Cost:
                if v_path is a complete tour (Hamiltonian Cycle):
                    Min_Cost = min(Min_Cost, v_cost) # Update global minimum
                else:
                    Q.insert((path=v_path, cost=v_cost, lower_bound=v_lower_bound))

    return Min_Cost

Q13. Floyd-Warshall Algorithm
- Explain core idea in one short sentence.
- Write the main function or pseudocode for generating the shortest path matrix.
Ans13 - Core Idea in one short sentence:
        It finds the shortest path between all pairs of vertices in a weighted graph by iteratively allowing intermediate vertices from a growing set $\{1, \dots, k\}$.

        Main Function/Pseudocode:
        function FLOYD_WARSHALL(W):
    n = number of vertices
    # Initialize distance matrix D with the input weight matrix W
    D = W 

    # The main triple loop: k is the intermediate vertex
    for k from 1 to n:
        # i is the start vertex
        for i from 1 to n:
            # j is the end vertex
            for j from 1 to n:
                # Relaxation Step: path i -> j is shorter if it goes through k
                # D[i][j] = min(D[i][j], D[i][k] + D[k][j])
                if D[i][k] + D[k][j] < D[i][j]:
                    D[i][j] = D[i][k] + D[k][j]

    return D # The shortest path matrix

