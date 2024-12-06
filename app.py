from flask import Flask, render_template, request
import random
import matplotlib.pyplot as plt
import os

from fp import calculate_fitness, genetic_algorithm

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None  # Inisialisasi default untuk hasil

    if request.method == "POST":
        try:
            # Ambil data dari form
            n = int(request.form["n"])
            queens_input = request.form["queens"]
            generations = int(request.form["generations"])
            mutation_rate = float(request.form["mutation_rate"])
            crossover_rate = float(request.form["crossover_rate"])
            max_mutations = int(request.form["max_mutations"])
            max_crossovers = int(request.form["max_crossovers"])

            queens = list(map(int, queens_input.split()))
            if len(queens) != n:
                return "Jumlah posisi awal ratu harus sama dengan ukuran papan (n).", 400

            # Konfigurasi awal populasi
            max_fitness = (n * (n - 1)) // 2
            population_size = 800
            population = []

            for _ in range(population_size):
                chromosome = queens[:]
                for i in range(len(chromosome)):
                    if chromosome[i] == 0:
                        chromosome[i] = random.randint(1, n)
                population.append(chromosome)

            # Jalankan algoritma genetika
            solution, generation_count = genetic_algorithm(
                population, max_fitness, generations, mutation_rate, crossover_rate,
                max_mutations, max_crossovers
            )

            # Simpan hasil
            if solution:
                save_chessboard(solution, n)
                result = {
                    "solved": True,
                    "solution": solution,
                    "generations": generation_count,
                    "time": 0.5  # Anda bisa menambahkan penghitungan waktu eksekusi
                }
            else:
                result = {
                    "solved": False,
                    "generations": generation_count
                }
        except Exception as e:
            return f"Terjadi kesalahan: {str(e)}", 500

    return render_template("index.html", result=result)

def save_chessboard(chromosome, n):
    board_colors = plt.imread("board_colors.png")  # Gambar untuk papan catur
    plt.figure(figsize=(8, 8))
    for i in range(n):
        plt.text(i + 0.5, n - chromosome[i] + 0.5, "♛", ha="center", va="center", fontsize=16, color="black")
    # Simpan gambar di folder 'static'
    plt.savefig("static/solution.png")  # Folder 'static' di dalam direktori aplikasi Flask
    plt.close()

if __name__ == "__main__":
    app.run(debug=True, port=5001)  
