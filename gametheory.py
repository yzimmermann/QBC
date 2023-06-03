import random
import numpy as np
import matplotlib.pyplot as plt

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

def evaluate_strategy(strategy, num_players, contraction_factor):
    # Simulate the classical game using the given strategy
    numbers = []
    for _ in range(num_players):
        measured_state = np.random.choice(range(len(strategy)), p=strategy)
        numbers.append(measured_state)
    average_number = calculate_average(numbers)
    closest_number = min(numbers, key=lambda x: abs(x - contraction_factor * average_number))
    return closest_number

def generate_initial_population(population_size, num_states):
    population = []
    for _ in range(population_size):
        strategy = np.random.random(num_states)
        strategy /= strategy.sum()  # Normalize the strategy vector
        population.append(strategy)
    return population

def evolve_population(population, num_players, elite_percentage, mutation_rate, contraction_factor):
    population_size = len(population)
    elite_size = int(elite_percentage * population_size)

    fitness_scores = []
    for strategy in population:
        fitness = evaluate_strategy(strategy, num_players, contraction_factor)
        fitness_scores.append(fitness)

    elite_indices = np.argsort(fitness_scores)[-elite_size:]
    elite_strategies = [population[i] for i in elite_indices]

    # Generate new strategies through mutation and crossover
    new_population = elite_strategies.copy()
    while len(new_population) < population_size:
        parent1, parent2 = random.choices(elite_strategies, k=2)
        child = crossover(parent1, parent2)
        mutated_child = mutate(child, mutation_rate)
        new_population.append(mutated_child)

    return new_population

def crossover(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def mutate(strategy, mutation_rate):
    # Perform mutation by randomly perturbing the strategy vector
    mutated_strategy = strategy.copy()
    for i in range(len(mutated_strategy)):
        if random.random() < mutation_rate:
            mutated_strategy[i] += random.uniform(-0.1, 0.1)
            mutated_strategy[i] = max(0, mutated_strategy[i])
    mutated_strategy = np.array(mutated_strategy)
    mutated_strategy /= mutated_strategy.sum()
    return mutated_strategy

# Parameters
population_size = 100
num_players = 100
num_states = 100
elite_percentage = 0.2
mutation_rate = 0.1
num_iterations = 100
contraction_factor = 0.5

population = generate_initial_population(population_size, num_states)

# Evolutionary algorithm loop
for iteration in range(num_iterations):
    best_strategy = None
    best_fitness = float('-inf')

    for strategy in population:
        fitness = evaluate_strategy(strategy, num_players, contraction_factor)
        if fitness > best_fitness:
            best_fitness = fitness
            best_strategy = strategy

    print(f"Iteration {iteration}: Best Fitness = {best_fitness}")

    population = evolve_population(population, num_players, elite_percentage, mutation_rate, contraction_factor)

print("Best strategy:", best_strategy)

# Plot the best strategy and mark the winning ket
plt.bar(range(len(best_strategy)), best_strategy)
plt.axvline(x=best_fitness, color='red')
plt.xlabel("State")
plt.ylabel("Probability")
plt.show()

