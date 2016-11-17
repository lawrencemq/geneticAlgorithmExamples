import random

from src.genetic_algorithm.search import GeneticAlgorithmSearch


class SmallNumber:
    def __init__(self, decimal_number: int):
        self.decimal_number = decimal_number

    @classmethod
    def from_binary_str(cls, binary: str):
        return SmallNumber(int(binary, 2))

    def __str__(self):
        binary = bin(self.decimal_number)[2:]
        while len(binary) < 6:
            binary = '0' + binary
        return binary

    def __repr__(self):
        return self.decimal_number


class SmallMinQuadraticGA(GeneticAlgorithmSearch):
    """
    Finds the minimum of a quadratic function via GA.
    Bounds must be within 0 and 63 - this means at most 6 bits

    To find the maximum of a quadratic, set find_min = False
    """

    def __init__(self, equation, boundary: (int, int), num_generations=5, population_size=4, find_min: bool=True):
        GeneticAlgorithmSearch.__init__(self, num_generations=num_generations)
        self._quadratic_eq = equation
        self._lower_bound, self._upper_bound = boundary
        self._population_size = population_size
        self.reverse_sort = find_min
        assert self._lower_bound < self._upper_bound

    def _generate_initial_population(self) -> [SmallNumber]:
        return [SmallNumber(random.randint(self._lower_bound, self._upper_bound)) for _ in range(self._population_size)]

    def _evaluate_chromosome(self, chromosome: SmallNumber):
        return self._quadratic_eq(chromosome.decimal_number)

    def _should_exclude(self, chromosome: SmallNumber) -> bool:
        return self._lower_bound > chromosome.decimal_number > self._upper_bound

    def _handle_mutation_in(self, chromosome: SmallNumber) -> SmallNumber:
        flip_bit = lambda x: '0' if x == '1' else '1'
        binary_generator = (flip_bit(bit) if random.randint(0, 100) < self.mutation_rate else bit
                            for bit in str(chromosome))
        return SmallNumber.from_binary_str(''.join(binary_generator))

    def _handle_crossover_between(self, chromosome1: SmallNumber, chromosome2: SmallNumber) -> SmallNumber:
        offspring_genes = ''.join(gene1 if random.randint(0, 100) < self.crossover_rate else gene2
                                  for gene1, gene2 in zip(str(chromosome1), str(chromosome2)))
        return SmallNumber.from_binary_str(offspring_genes)


if __name__ == '__main__':
    ga = SmallMinQuadraticGA(equation=lambda x: (x**2 - 10*x + 5), boundary=(0, 63))
    ga.run_search()
    best_estimate = ga.get_result()
    print('Best Estimated Minimum {}'.format(best_estimate.decimal_number))
