import random
import matplotlib.pyplot as plt
import numpy as np
import time

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

def calculate_score(generations, time_taken, w1=1, w2=1):
    return w1 * generations + w2 * time_taken

if __name__ == "__main__":
    best_solutions = []
    n = None
    max_fitness = None
    population_size = 800
    max_generations = None
    mutation_rate = None
    crossover_rate = None
    initial_queens = None

    while True:
        user_input = int(input("Input 1 untuk menginput data baru atau 0 untuk keluar: "))
        if user_input == 0:
            break

        if user_input == 1:
            if n is None:
                n = int(input("Masukkan ukuran papan catur (n): "))
                initial_queens = list(map(int, input(f"Masukkan posisi awal ratu (1 hingga {n}, dipisahkan spasi): ").split()))

                if len(initial_queens) != n:
                    print(f"Jumlah kolom harus tepat {n}.")
                    continue

                max_fitness = (n * (n - 1)) // 2
                max_generations = int(input("Masukkan jumlah iterasi maksimum: "))
                mutation_rate = float(input("Masukkan probabilitas mutasi (0.0 hingga 1.0): "))
                crossover_rate = float(input("Masukkan probabilitas crossover (0.0 hingga 1.0): "))
            else:
                print(f"Menggunakan ukuran papan catur (n): {n}")
                print(f"Menggunakan posisi awal ratu: {initial_queens}")

            max_mutations = int(input("Masukkan jumlah mutasi maksimum: "))
            max_crossovers = int(input("Masukkan jumlah crossover maksimum: "))

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
                print(f"Solved in {generations} generations!")
                print(f"Solution: {solution}")
                best_solutions.append((solution, generations, elapsed_time, max_mutations, max_crossovers))
            else:
                print(f"Unable to solve within {max_generations} generations.")
                best_solutions.append((None, max_generations, elapsed_time, max_mutations, max_crossovers))

   
    if best_solutions:
        best_solution = min([s for s in best_solutions if s[0] is not None], key=lambda x: x[1], default=None)
        if best_solution:
            print("\nIterasi terbaik:")
            print(f"Solusi: {best_solution[0]}")
            print(f"Ditemukan dalam {best_solution[1]} generasi dengan waktu {best_solution[2]:.2f} detik.")
            print(f"Jumlah mutasi maksimum: {best_solution[3]}, Jumlah crossover maksimum: {best_solution[4]}")
        else:
            print("\nTidak ada solusi yang ditemukan.")
        
        print("\nRingkasan Hasil Percobaan:")
        for idx, (sol, gen, time_taken, mutations, crossovers) in enumerate(best_solutions):
            status = "Solved" if sol else "Not Solved"
            print(f"Percobaan {idx + 1}: {status} dalam {gen} generasi, waktu {time_taken:.2f} detik, mutasi maksimum: {mutations}, crossover maksimum: {crossovers}")
        
        
        solved_solutions = [s for s in best_solutions if s[0] is not None]
        if solved_solutions:
            best_generation_solution = min(solved_solutions, key=lambda x: x[1])
            fastest_solution = min(solved_solutions, key=lambda x: x[2])
            best_scored_solution = min(
                solved_solutions, 
                key=lambda x: calculate_score(x[1], x[2], w1=1, w2=1)
            )
            print("\nKesimpulan:")
            print(f"Solusi dengan generasi terbaik:")
            print(f"  Solusi: {best_generation_solution[0]} dalam {best_generation_solution[1]} generasi")
            print(f"  Waktu: {best_generation_solution[2]:.2f} detik")
            
            print(f"\nSolusi tercepat:")
            print(f"  Solusi: {fastest_solution[0]} ditemukan dalam {fastest_solution[2]:.2f} detik")
            print(f"  Generasi: {fastest_solution[1]}")

            print(f"\nSolusi terbaik berdasarkan skor (w1=1, w2=1):")
            print(f"  Solusi: {best_scored_solution[0]} dalam {best_scored_solution[1]} generasi")
            print(f"  Waktu: {best_scored_solution[2]:.2f} detik")
            print(f"  Skor: {calculate_score(best_scored_solution[1], best_scored_solution[2])}")
        else:
            print("\nTidak ada solusi yang berhasil ditemukan untuk dibandingkan.")
    else:
        print("Tidak ada percobaan yang dilakukan.")
