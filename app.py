from flask import Flask, render_template, request, jsonify
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Your genetic algorithm functions here (the same as in your code)
def calculate_fitness(chromosome, max_fitness):
    n = len(chromosome)

    horizontal_collisions = sum([chromosome.count(queen) - 1 for queen in chromosome]) // 2

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

def crossover(x, y, crossover_rate, max_crossovers, current_crossovers):
    if current_crossovers[0] < max_crossovers:
        n = len(x)
        child = [0] * n
        for i in range(n):
            child[i] = x[i] if random.random() > 0.5 else y[i]
        
        current_crossovers[0] += 1
        return child
    return x

def mutate(chromosome, mutation_rate, max_mutations, current_mutations):
    if current_mutations[0] < max_mutations:
        n = len(chromosome)
        index = random.randint(0, n - 1)
        value = random.randint(1, n)
        chromosome[index] = value
        current_mutations[0] += 1
    
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
    board_colors = np.zeros((n, n, 3))
    board = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                board_colors[i, j] = [1, 0.87, 0.67]
            else:
                board_colors[i, j] = [0.65, 0.42, 0.27]

    for i in range(n):
        board[chromosome[i] - 1][i] = 1

    plt.clf()
    plt.imshow(board_colors, extent=[0, n, 0, n])
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.grid(color="black", linestyle="-", linewidth=1.5)
    for i in range(n):
        plt.text(i + 0.5, n - chromosome[i] + 0.5, "♛", ha="center", va="center", fontsize=16, color="black")
    plt.title(f"Generation: {generation}")
    plt.pause(0.1)

def genetic_algorithm(population, max_fitness, max_generations, mutation_rate, crossover_rate, max_mutations, max_crossovers):
    
    plt.ion()
    generation = 1
    
    current_mutations = [0]
    current_crossovers = [0]
    
    best_fitness = 0
    stagnation_count = 0
    
    while generation <= max_generations:
        fitness_values = [calculate_fitness(chromosome, max_fitness) for chromosome in population]
        probabilities = [fitness / max_fitness for fitness in fitness_values]
        sorted_population = sorted(zip(fitness_values, population), key=lambda x: x[0], reverse=True)

        best_chromosome = sorted_population[0][1]
        current_best_fitness = sorted_population[0][0]
        print_chessboard(best_chromosome, generation)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            stagnation_count = 0
        else:
            stagnation_count += 1

        if current_best_fitness == max_fitness:
            plt.ioff()
            plt.show()
            return best_chromosome, generation
        
        if stagnation_count > 10:
            max_mutations *= 2
            max_crossovers *= 2
            stagnation_count = 0

        new_population = [best_chromosome]
        while len(new_population) < len(population):
            parent1 = roulette_wheel_selection(population, probabilities)
            parent2 = roulette_wheel_selection(population, probabilities)
            
            child = crossover(parent1, parent2, crossover_rate, max_crossovers, current_crossovers)
            child = mutate(child, mutation_rate, max_mutations, current_mutations)
            
            new_population.append(child)

        population = new_population
        generation += 1

    plt.ioff()
    plt.show()
    return None, max_generations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    n = int(request.form['n'])
    initial_queens = list(map(int, request.form['initial_queens'].split()))
    max_generations = int(request.form['max_generations'])
    mutation_rate = float(request.form['mutation_rate'])
    crossover_rate = float(request.form['crossover_rate'])
    max_mutations = int(request.form['max_mutations'])
    max_crossovers = int(request.form['max_crossovers'])
    
    max_fitness = (n * (n - 1)) // 2
    population_size = 800
    
    population = []
    for _ in range(population_size):
        chromosome = initial_queens[:]
        for i in range(len(chromosome)):
            if chromosome[i] == 0:
                chromosome[i] = random.randint(1, n)
        population.append(chromosome)

    start_time = time.time()
    solution, generations = genetic_algorithm(
        population, max_fitness, max_generations, mutation_rate, crossover_rate,
        max_mutations, max_crossovers
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    if solution:
        status = 'Solved'
    else:
        status = 'Not Solved'

    return jsonify({
        'status': status,
        'solution': solution,
        'generations': generations,
        'time': round(elapsed_time, 2),
        'mutation_rate': max_mutations,
        'crossover_rate': max_crossovers
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)