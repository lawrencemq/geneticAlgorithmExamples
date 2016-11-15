import random
from abc import abstractmethod


class GeneticAlgorithmSearch:

    def __init__(self, num_generations: int=100):
        self.keep_best = True
        self.num_generations = num_generations
        self.verbose_print_every = 10  # Display where we are every 10%
        self.verbose = True
        self._current_population = []
        self._best_so_far = None
        self.crossover_rate = 50  # 50%
        self.mutation_rate = 10  # 10%
        self.reverse_sort = False

    @abstractmethod
    def _generate_initial_population(self):
        raise NotImplementedError

    @abstractmethod
    def _handle_crossover_between(self, chromosome1, chromosome2):
        raise NotImplementedError

    @abstractmethod
    def _handle_mutation_in(self, chromosome):
        raise NotImplementedError

    @abstractmethod
    def _should_exclude(self, chromosome):
        raise NotImplementedError

    @abstractmethod
    def _evaluate_chromosome(self, chromosome):
        raise NotImplementedError

    def __create_probabalistic_population_for_pick(self):
        """Assumes population is already sorted by performance - best last"""
        to_return = []
        for position, chromosome in enumerate(self._current_population):
            to_return.extend([chromosome]*position)
        return to_return

    def __verbose(self, *to_print):
        if self.verbose:
            print(*to_print, sep=" ")

    def run_search(self):
        self._current_population = self._generate_initial_population()

        for i in range(self.num_generations):
            self.__verbose('Starting generation {}'.format(i))
            
            # Evaluate
            self._current_population.sort(key=self._evaluate_chromosome)
            self._best_so_far = self._current_population[-1]
            self.__verbose('\tBest score so far = {}'.format(self._evaluate_chromosome(self._best_so_far)))

            # Creating new population
            new_population = []

            # Copy best over if needed
            if self.keep_best:
                new_population.append(self._best_so_far)

            # Filling the rest
            probabilistic_population_for_mating = self.__create_probabalistic_population_for_pick()
            while len(new_population) < len(self._current_population):
                parent1 = random.choice(probabilistic_population_for_mating)
                parent2 = random.choice(probabilistic_population_for_mating)

                # Performing crossover
                child = self._handle_crossover_between(parent1, parent2)

                # Performing mutation
                child = self._handle_mutation_in(child)

                # Ensuring child is good
                if self._should_exclude(child):
                    continue

                new_population.append(child)

            self._current_population = new_population

    def get_result(self):
        return self._best_so_far
