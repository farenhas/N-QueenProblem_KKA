import random
import matplotlib.pyplot as plt
import numpy as np

def calculate_fitness(chromosome, max_fitness):
    n = len(chromosome)

    # Calculate horizontal collisions
    horizontal_collisions = sum([chromosome.count(queen) - 1 for queen in chromosome]) // 2

    # Calculate diagonal collisions
    left_diagonal = [0] * (2 * n - 1)
    right_diagonal = [0] * (2 * n - 1)
    for i in range(n):
        left_diagonal[i + chromosome[i] - 1] += 1
        right_diagonal[n - i + chromosome[i] - 2] += 1

    diagonal_collisions = 0
    for count in left_diagonal + right_diagonal:
        if count > 1:
            diagonal_collisions += count - 1

    return max_fitness - (horizontal_collisions + diagonal_collisions)

def crossover(x, y, crossover_rate):
    if random.random() > crossover_rate:
        return x
    n = len(x)
    child = [0] * n
    for i in range(n):
        child[i] = x[i] if random.random() > 0.5 else y[i]
    return child

def mutate(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        n = len(chromosome)
        index = random.randint(0, n - 1)
        value = random.randint(1, n)
        chromosome[index] = value
    return chromosome

def roulette_wheel_selection(population, probabilities):
    total = sum(probabilities)
    r = random.uniform(0, total)
    upto = 0
    for chromosome, probability in zip(population, probabilities):
        if upto + probability >= r:
            return chromosome
        upto += probability
    return population[-1]

def print_chessboard(chromosome, generation):
    n = len(chromosome)
    board_colors = np.zeros((n, n, 3))  # RGB colors for the board
    board = np.zeros((n, n))

    # Create chessboard pattern
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                board_colors[i, j] = [1, 0.87, 0.67]  # Light color (like beige)
            else:
                board_colors[i, j] = [0.65, 0.42, 0.27]  # Dark color (like brown)

    # Place queens
    for i in range(n):
        board[chromosome[i] - 1][i] = 1

    plt.clf()  # Clear the previous figure
    plt.imshow(board_colors, extent=[0, n, 0, n])
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.grid(color="black", linestyle="-", linewidth=1.5)  # Add grid lines
    for i in range(n):
        plt.text(i + 0.5, n - chromosome[i] + 0.5, "♛", ha="center", va="center", fontsize=16, color="black")
    plt.title(f"Generation: {generation}")
    plt.pause(1)  # Pause for 1 second

def genetic_algorithm(population, max_fitness, max_generations, mutation_rate, crossover_rate):
    plt.ion()  # Enable interactive mode for live updates
    generation = 1
    while generation <= max_generations:
        fitness_values = [calculate_fitness(chromosome, max_fitness) for chromosome in population]
        probabilities = [fitness / max_fitness for fitness in fitness_values]
        sorted_population = sorted(zip(fitness_values, population), key=lambda x: x[0], reverse=True)

        best_chromosome = sorted_population[0][1]
        print_chessboard(best_chromosome, generation)

        if sorted_population[0][0] == max_fitness:
            plt.ioff()  # Disable interactive mode
            return best_chromosome, generation

        new_population = [best_chromosome]
        while len(new_population) < len(population):
            parent1 = roulette_wheel_selection(population, probabilities)
            parent2 = roulette_wheel_selection(population, probabilities)
            child = crossover(parent1, parent2, crossover_rate)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        generation += 1

    plt.ioff()  # Disable interactive mode
    return None, max_generations

if __name__ == "__main__":
    n = int(input("Masukkan ukuran papan catur (n): "))
    print(f"Masukkan posisi awal ratu (1 hingga {n}, dipisahkan spasi):")
    initial_queens = list(map(int, input().split()))

    if len(initial_queens) != n:
        print(f"Jumlah kolom harus tepat {n}.")
        exit()

    max_fitness = (n * (n - 1)) // 2  # Maximum fitness for N-Queens problem
    population_size = 800

    # Parameter Genetic Algorithm
    max_generations = int(input("Masukkan jumlah iterasi maksimum: "))
    mutation_rate = float(input("Masukkan probabilitas mutasi (0.0 hingga 1.0): "))
    crossover_rate = float(input("Masukkan probabilitas crossover (0.0 hingga 1.0): "))

    # Generate initial population
    population = []
    for _ in range(population_size):
        chromosome = initial_queens[:]
        for i in range(len(chromosome)):
            if chromosome[i] == 0:  # If a queen is not placed
                chromosome[i] = random.randint(1, n)
        population.append(chromosome)

    solution, generations = genetic_algorithm(
        population, max_fitness, max_generations, mutation_rate, crossover_rate
    )

    if solution:
        print(f"Solved in {generations} generations!")
        print(f"Solution: {solution}")
    else:
        print(f"Unable to solve within {max_generations} generations.")
