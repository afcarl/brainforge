from .abstract_population import Population


class MemeticAlgorithm(Population):

    def update_individual(self, ind, *fitness_args, **fitness_kw):
        genome = self.individuals[ind]
        updated_genome, *fitnesses = self.fitnesses(genome, *fitness_args, **fitness_kw)
        self.fitnesses[ind] = fitnesses
        self.grades[ind] = self.grade_function(self.fitnesses[ind])
        self.individuals[ind] = updated_genome
