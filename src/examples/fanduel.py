import csv
import random

from src.genetic_algorithm.search import GeneticAlgorithmSearch


class FanDuelPlayer:
    def __init__(self, name: str, salary: int, projected_points: float, team: str, position: str, injured: bool):
        self.name = name
        self.salary = salary
        self.projected_points = projected_points
        self.team = team
        self.position = position
        self.injured = injured

    def __repr__(self):
        return '{} ({})'.format(self.name, self.position)


class Lineup:
    def __init__(self, players: [FanDuelPlayer]):
        self.players = players

    def count(self, player: FanDuelPlayer):
        return self.players.count(player)

    def index(self, player: FanDuelPlayer):
        return self.players.index(player)

    def __len__(self):
        return len(self.players)

    def __iter__(self):
        return iter(self.players)

    def __getitem__(self, item):
        return self.players[item]

    def __setitem__(self, key, value):
        self.players[key] = value


def _read_fanduel_data(filename: str) -> [FanDuelPlayer]:
    """
    Reads the csv file from FanDuel and returns a dictionary that
    :param filename: CSV file downloaded from FanDuel
    :return: List of FanDuelPalayers and their information.
    """
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return [
            FanDuelPlayer(
                name=" ".join([row['First Name'], row['Last Name']]),
                salary=int(row['Salary']),
                projected_points=float(row['FPPG']),
                team=row['Team'],
                position=row['Position'],
                injured=row['Injury Indicator'] in ['', 'P']  # Players with no injury or who are probable are "healthy"
            )
            for row in reader
            ]


class FanDuelFootballGA(GeneticAlgorithmSearch):

    def __init__(self, filename: str, salary_cap=60000, num_generations=100, population_size=20):
        GeneticAlgorithmSearch.__init__(self, num_generations=num_generations)
        self._population_size = population_size
        self._salary_cap = salary_cap
        self._all_players = _read_fanduel_data(filename)
        self._qb_list = [x for x in self._all_players if x.position == 'QB']
        self._rb_list = [x for x in self._all_players if x.position == 'RB']
        self._wr_list = [x for x in self._all_players if x.position == 'WR']
        self._te_list = [x for x in self._all_players if x.position == 'TE']
        self._d_list = [x for x in self._all_players if x.position == 'D']
        self._k_list = [x for x in self._all_players if x.position == 'K']
        self.position_to_player_list_map = {
            'QB': self._qb_list,
            'RB': self._rb_list,
            'WR': self._wr_list,
            'TE': self._te_list,
            'D': self._d_list,
            'K': self._k_list,
        }

    def _generate_initial_population(self) -> [Lineup]:
        population = []
        for _ in range(self._population_size):
            lineup = [random.choice(self._qb_list)]
            lineup.extend(random.sample(self._rb_list, 2))
            lineup.extend(random.sample(self._wr_list, 3))
            lineup.append(random.choice(self._te_list))
            lineup.append(random.choice(self._d_list))
            lineup.append(random.choice(self._k_list))
            population.append(Lineup(lineup))
        return population

    def _evaluate_chromosome(self, lineup: Lineup) -> float:
        return sum(player.projected_points for player in lineup)

    @staticmethod
    def __find_player_to_replace(lineup: Lineup, position: str) -> FanDuelPlayer:
        players_of_position = {player for player in lineup if player.position == position}
        for player in players_of_position:
            if lineup.count(player) > 1:
                return player
        raise ValueError("Unable to find duplicate players in lineup!")

    def _handle_crossover_between(self, chromosome1: Lineup, chromosome2: Lineup) -> Lineup:
        crossover_index = random.randint(0, len(chromosome1))
        new_lineup = Lineup(chromosome1[:crossover_index] + chromosome2[crossover_index:])

        # checking running backs
        while len({player for player in new_lineup if player.position == 'RB'}) < 2:
            self.__replace_duplicate_player_in_lineup(new_lineup, 'RB')

        # checking wide receivers
        while len({player for player in new_lineup if player.position == 'WR'}) < 3:
            self.__replace_duplicate_player_in_lineup(new_lineup, 'WR')

        return new_lineup

    def __replace_duplicate_player_in_lineup(self, new_lineup: Lineup, position: str):
        # finding player to replace
        to_replace = self.__find_player_to_replace(new_lineup, position)
        index_of_player = new_lineup.index(to_replace)
        replacement_player = self.__randomly_choose_player_not_in(new_lineup, position)
        new_lineup[index_of_player] = replacement_player

    def _handle_mutation_in(self, lineup: Lineup) -> Lineup:
        for index, player in enumerate(lineup):
            if random.randint(0, 100) < self.mutation_rate:
                # picking new random player of same position not already in lineup
                lineup[index] = self.__randomly_choose_player_not_in(lineup, player.position)
        return lineup

    def __randomly_choose_player_not_in(self, lineup: Lineup, position: str):
        return random.choice(list(set(self.position_to_player_list_map[position]).difference({p for p in lineup if p.position == position})))

    def _should_exclude(self, lineup: Lineup) -> bool:
        """
        Ensures that the lineups are accurate, that there are no duplicate players, and that the salary cap is met.
        :param lineup:
        :return:
        """
        def count_position(position: str) -> bool:
            return sum(player.position == position for player in lineup)

        return not (
            sum(player.salary for player in lineup) < self._salary_cap
            and count_position('QB') == 1
            and count_position('RB') == 2
            and count_position('WR') == 3
            and count_position('TE') == 1
            and count_position('D') == 1
            and count_position('K') == 1
            and len(set(lineup)) == 9
            and max(sum(1 for player in lineup if player.team == team) for team in {player.team for player in lineup}) <= 3
        )


if __name__ == '__main__':
    ga = FanDuelFootballGA('../../FanDuel-NFL-2016-11-17-16937-players-list.csv')
    ga.run_search()
    best_lineup = ga.get_result()
    print('Best Lineup ${}'.format(sum(player.salary for player in best_lineup)))
    for player in best_lineup:
        print(player)