from .abstract_population import Population


class DifferentialEvolution(Population):

    def update_individual(self, ind, *fitness_args, **fitness_kw):
        genome = self.individuals[ind]
        self.fitnesses[ind] = self.fitnesses(genome, *fitness_args, **fitness_kw)
        self.grades[ind] = self.grade_function(self.fitnesses[ind])
