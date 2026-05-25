import os
import time
import random
import pickle
import signal
import logging
import datetime
import subprocess
import concurrent.futures

from llm_caller import LLMCaller
from utils import *

# Structure of individual
class Individual:

    def __init__(
        self,
        iter_id,
        resp_id,
        code,
        code_path,
        stdout_path,
        idea="",
        exec_success=None,
        fitness=None,
        solver_info=None,
    ):
        self.iter_id = iter_id
        self.resp_id = resp_id
        self.code = code
        self.code_path = code_path
        self.stdout_path = stdout_path
        self.idea = idea
        self.exec_success = exec_success
        self.fitness = fitness
        self.solver_info = solver_info

    # print the individual info
    def __repr__(self):
        code_preview = self.code[:18] + " ... " + self.code[-12:] if len(self.code) > 30 else self.code
        idea_preview = self.idea[:18] + " ... " + self.idea[-12:] if len(self.idea) > 30 else self.idea
        return (
            f"Individual(\n"
            f"  iter_id={self.iter_id},\n"
            f"  resp_id={self.resp_id},\n"
            f"  code={code_preview!r},\n"
            f"  idea={idea_preview!r},\n"
            f"  code_path={self.code_path!r},\n"
            f"  stdout_path={self.stdout_path!r},\n"
            f"  exec_success={self.exec_success},\n"
            f"  fitness={self.fitness}\n"
            f"  solver_info={self.solver_info}\n"
            f")"
        )

# Structure of population
class Population:

    def __init__(
        self,
        individuals=[],
        ):
        self.individuals = individuals

    def add(self, indv):
        self.individuals.append(indv)

    def __iter__(self):
        return iter(self.individuals)

    def __len__(self):
        return len(self.individuals)

    def __repr__(self):
        individual_info = ""
        for i, indv in enumerate(self.individuals):
            individual_info += f"  Indv {i}:\n"
            individual_info += f"    iter_id={indv.iter_id}\n"
            individual_info += f"    resp_id={indv.resp_id}\n"
            individual_info += f"    fitness={indv.fitness}\n"
        
        return (
            f"Population(\n"
            f"  size={self.__len__()}\n"
            f"  individuals=[\n"
            f"{individual_info}"
            f"  ]\n"
            f")"
        )

    # Select the top pop_size individuals
    def select_top(self, pop_size):
        self.individuals.sort(key=lambda x: x.fitness)
        self.individuals = self.individuals[:pop_size]
        return self
    
    # Select two parents for crossover
    def select_parent(self):

        assert len(self.individuals) >= 2, "At least two individuals are required to select parents."
        # ranks as probabilities
        self.individuals.sort(key=lambda x: x.fitness)  # sort by fitness
        ranks = [i for i in range(len(self.individuals))]
        probs = [1 / (rank + 1 + len(self.individuals)) for rank in ranks]
        probs = [prob / sum(probs) for prob in probs]  # normalize the probabilities
        
        id_1, id_2 = random.choices(ranks, weights=probs, k=2)
        while id_1 == id_2:  # until two different parents are selected
            id_1, id_2 = random.choices(ranks, weights=probs, k=2)
                
        return self.individuals[min(id_1, id_2)], self.individuals[max(id_1, id_2)]  # the first parent is better

# FormuEvo evolution pipeline
class FormuEvo:

    def __init__(
        self,
        config
        ):

        self.config = config
        self.iter_id = 0
        self.resp_id = 0

    def read_prompt(self):

        # read config
        self.problem = self.config["problem"]
        self.pop_size = self.config["pop_size"]
        self.max_iter = self.config["max_iter"]
        self.mut_rate = self.config["mut_rate"]
        self.mem_rate = self.config["mem_rate"]
        self.timeout = self.config["timeout"]
        try:
            self.cpu_ls = list(os.sched_getaffinity(0))
            self.cpu_ls = get_physical_cores(self.cpu_ls)
        except:
            self.cpu_ls = None

        # file and directory path
        self.run_file = f"problems/{self.problem}/gpt.py"  # the file to run the code
        self.eval_file = f"problems/{self.problem}/eval.py"  # the file to evaluate the code
        self.out_dir = f"outputs/{self.problem}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} ({self.config['model'].split('/')[-1]})"  # the directory to store the output
        os.makedirs(self.out_dir, exist_ok=True)  # make directory

        # set up logging to file
        log_handler = logging.FileHandler(os.path.join(self.out_dir, 'evolution.log'))
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(log_handler)

        # log configuration
        logging.info(f"Problem: {self.problem}")
        logging.info(f"Population size: {self.pop_size}")
        logging.info(f"Total iterations: {self.max_iter}")
        logging.info(f"Mutation rate: {self.mut_rate}")
        logging.info(f"Memory rate: {self.mem_rate}")
        logging.info(f"Timeout limit: {self.timeout}s")
        logging.info(f"Num training: {self.config.get('num_train', 100)}")
        logging.info(f"Num validate: {self.config.get('num_val', 100)}")
        logging.info(f"Model: {self.config['model']}")
        logging.info(f"Provider: {self.config['provider']}")
        logging.info(f"CPU cores: {self.cpu_ls if self.cpu_ls else 'None'} \n")

        # initialize LLM agent
        self.llm_caller = LLMCaller(
            provider=self.config["provider"],
            model=self.config["model"],
            )

        # load seed function and template
        with open(os.path.join("problems", self.problem, "prompts", "seed_func.txt"), 'r') as f:
            self.seed_func = f.read()  # seed function
        with open(os.path.join("problems", self.problem, "prompts", "temp_func.txt"), 'r') as f:
            self.temp_func = f.read()  # function template

        # load initialization prompt
        with open(os.path.join("prompts", "initialization.txt"), 'r') as f:
            self.system_initialization = f.read()  # system prompt for initialization
        with open(os.path.join("problems", self.problem, "prompts", "description.txt"), 'r') as f:
            self.problem_description = f.read()  # problem description
        
        # load crossover prompt
        with open(os.path.join("prompts", "crossover", "intersection.txt"), 'r') as f:
            self.system_intersection = f.read()  # system prompt for intersection
        with open(os.path.join("prompts", "crossover", "complement.txt"), 'r') as f:
            self.system_complement = f.read()  # system prompt for complement
        with open(os.path.join("prompts", "crossover", "hybrid.txt"), 'r') as f:
            self.system_hybrid = f.read()  # system prompt for hybrid
        with open(os.path.join("prompts", "crossover", "min_violation.txt"), 'r') as f:
            self.system_min_violation = f.read()  # system prompt for min violation
        self.system_crossover = [
            self.system_intersection,
            self.system_complement,
            self.system_hybrid,
            self.system_min_violation
            ]

        # load mutation prompt
        with open(os.path.join("prompts", "mutation", "general.txt"), 'r') as f:
            self.system_general = f.read()  # system prompt for general mutation
        with open(os.path.join("prompts", "mutation", "exploratory.txt"), 'r') as f:
            self.system_exploratory = f.read()  # system prompt for exploratory mutation
        with open(os.path.join("prompts", "mutation", "polyhedral.txt"), 'r') as f:
            self.system_polyhedral = f.read()  # system prompt for polyhedral mutation
        with open(os.path.join("prompts", "mutation", "searching.txt"), 'r') as f:
            self.system_searching = f.read()  # system prompt for searching mutation
        self.system_mutation = [
            self.system_general,
            self.system_exploratory,
            self.system_polyhedral,
            self.system_searching
            ]
        
        # load repair prompt
        with open(os.path.join("prompts", "repair.txt"), 'r') as f:
            self.system_repair = f.read()  # system prompt for repair

        # load diagnosis prompt
        with open(os.path.join("prompts", "diagnosis", "crossover.txt"), 'r') as f:
            self.system_diagnosis_crossover = f.read()  # system prompt for diagnosis of crossover
        with open(os.path.join("prompts", "diagnosis", "mutation.txt"), 'r') as f:
            self.system_diagnosis_mutation = f.read()  # system prompt for diagnosis of mutation

        # load memory prompt
        with open(os.path.join("prompts", "memory", "reflection.txt"), 'r') as f:
            self.system_reflection = f.read()  # system prompt for reflection
        with open(os.path.join("prompts", "memory", "curation.txt"), 'r') as f:
            self.system_curation = f.read()  # system prompt for curation

        self.problem_description = self.problem_description.strip()
        self.user_initialization = "Problem Description: \n" + self.problem_description + "\n\n" + \
            "Python Template: \n" + self.seed_func  # user prompt for initialization (description + seed function)
        self.input_initialization = [
            {"role": "system",
            "content": self.system_initialization},
            {"role": "user",
            "content": self.user_initialization}
        ]
        self.memory_library = ""  # memory library (empty at start)

    def seed_evaluation(self):

        # Read the seed function
        logging.info("Evaluating seed function...")
        seed_code = extract_code(self.seed_func)  # read the seed code
        # logging.info("Seed function code: \n" + seed_code)
        self.seed_indv = Individual(  # record the seed code as iter 0, resp 0
            iter_id=self.iter_id,  # 0
            resp_id=self.resp_id,  # 0
            code=seed_code,
            code_path=f"iter{self.iter_id}_code{self.resp_id}.txt",
            stdout_path=f"iter{self.iter_id}_stdout{self.resp_id}.txt",
            idea="Example for the Gurobi model.",
            exec_success=None,  # no evaluated yet
            fitness=None,  # not evaluated yet
            solver_info=None,
        )
        # initialize the population with the seed function
        self.valid_pop = Population(individuals=[self.seed_indv])
        self.resp_id += 1  # update the response id

        # Evaluate the seed function
        self.eval_population(pop=self.valid_pop)

        # Check seed function
        if self.valid_pop.individuals[0].exec_success == True and self.valid_pop.individuals[0].fitness is not None:
            pass  # seed function evaluation success
        else:
            logging.info("Seed function evaluation failed! Exiting...")
            exit(1)

    def init_population(self):

        logging.info(f"Initialization started!")

        # Make the population to pop_size (no evaluated yet)
        invalid_pop = Population(individuals=[])
        while len(self.valid_pop) < self.pop_size:

            # add invalid individuals to new generation
            num_invalid = len(invalid_pop)
            input_repair_ls = []
            for indv in invalid_pop.individuals:
                # logging.info(f"Iteration {indv.iter_id}, Repair code {indv.resp_id}")
                with open(os.path.join(self.out_dir, indv.stdout_path), 'r',encoding="utf-8") as f:  # read the stdout file
                    stdout_str = f.read()
                    traceback_info = filter_traceback(stdout_str)
                user_repair = "Problem Description: \n" + self.problem_description + "\n\n" + \
                    "Python Template: \n" + self.temp_func + "\n\n" + \
                    "Failed Code: \n" + indv.code + "\n\n" + \
                    "Corresponding Idea: \n" + indv.idea + "\n\n" + \
                    "Traceback Info: \n" + traceback_info
                input_repair_ls.append([
                    {"role": "system",
                    "content": self.system_repair},
                    {"role": "user",
                    "content": user_repair}
                ])

            # generate new individuals in parallel for initialization
            self.new_pop = Population(individuals=[])
            input_initialization_ls = [self.input_initialization] * (self.pop_size - len(self.valid_pop))
            new_outputs = self.llm_caller.call_batch(input_repair_ls + input_initialization_ls)
            # write response to file and add to new_pop
            for new_output in new_outputs:
                self.write_new_pop(new_output=new_output, new_pop=self.new_pop)
                self.resp_id += 1  # update the response id

            # Collective evaluation of the new population
            if len(self.new_pop) > 0:
                self.eval_population(pop=self.new_pop)

            # Update the valid population
            invalid_pop = Population(individuals=[])
            for i, indv in enumerate(self.new_pop.individuals):
                if indv.exec_success == True and indv.fitness is not None:
                    self.valid_pop.add(indv=indv)
                else:
                    if i >= num_invalid and indv.code is not None:
                        invalid_pop.add(indv=indv)  # record invalid individuals
            
        # Select the top pop_size individuals
        self.valid_pop = self.valid_pop.select_top(pop_size=self.pop_size)
        best_indv = self.valid_pop.individuals[0]
        logging.info(f"Best: Iter {best_indv.iter_id}, Code {best_indv.resp_id}, Fitness: {best_indv.fitness:.4f}")
        logging.info(f"Initialization finished!\n")
        
        # Update iteration info
        self.iter_id += 1  # update the iteration id
        self.resp_id = 0  # reset the response id

    def step_evolution(self):
        
        logging.info(f"Evolution iteration {self.iter_id} started!")

        input_reflection_ls = []
        invalid_pop = Population(individuals=[])
        while len(self.valid_pop) < 2 * self.pop_size:

            # add invalid individuals to new generation
            num_invalid = len(invalid_pop)
            input_repair_ls = []
            for indv in invalid_pop.individuals:
                # logging.info(f"Iteration {indv.iter_id}, Repair code {indv.resp_id}")
                with open(os.path.join(self.out_dir, indv.stdout_path), 'r',encoding="utf-8") as f:  # read the stdout file
                    stdout_str = f.read()
                    traceback_info = filter_traceback(stdout_str)
                user_repair = "Problem Description: \n" + self.problem_description + "\n\n" + \
                    "Python Template: \n" + self.temp_func + "\n\n" + \
                    "Failed Code: \n" + indv.code + "\n\n" + \
                    "Corresponding Idea: \n" + indv.idea + "\n\n" + \
                    "Traceback Info: \n" + traceback_info
                input_repair_ls.append([
                    {"role": "system",
                    "content": self.system_repair},
                    {"role": "user",
                    "content": user_repair}
                ])

            # generate diagnosis prompt
            input_diagnosis_ls, parents_indv_ls = [], []
            while len(self.valid_pop) + len(input_diagnosis_ls) < 2 * self.pop_size:
                # crossover diagnosis
                # select parents
                parent_1, parent_2 = self.valid_pop.select_parent()
                # diagnosis user prompt
                user_diagnosis_crossover = (
                    f"Problem Description: \n{self.problem_description}\n\n"
                    f"Parent 1: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                    f"RunTime: \n{str(parent_1.fitness)}\nSolver Information: \n{parent_1.solver_info}\n\n"
                    f"Parent 2: \nCode: \n{parent_2.code}\nIdea: \n{parent_2.idea}\n"
                    f"RunTime: \n{str(parent_2.fitness)}\nSolver Information: \n{parent_2.solver_info}\n\n"
                    f"Memory Library: \n{self.memory_library}"
                ) if random.random() < self.mem_rate else (
                    f"Problem Description: \n{self.problem_description}\n\n"
                    f"Parent 1: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                    f"RunTime: \n{str(parent_1.fitness)}\nSolver Information: \n{parent_1.solver_info}\n\n"
                    f"Parent 2: \nCode: \n{parent_2.code}\nIdea: \n{parent_2.idea}\n"
                    f"RunTime: \n{str(parent_2.fitness)}\nSolver Information: \n{parent_2.solver_info}\n\n"
                )

                # diagnosis input
                input_diagnosis_ls.append([
                    {"role": "system",
                    "content": self.system_diagnosis_crossover},
                    {"role": "user",
                    "content": user_diagnosis_crossover}
                ])
                parents_indv_ls.append(("crossover", parent_1, parent_2))
                # mutation
                if random.random() < self.mut_rate:  # consider mutation rate
                    # diagnosis user prompt
                    user_diagnosis_mutation = (
                        f"Problem Description: \n{self.problem_description}\n\n"
                        f"Current Model: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                        f"RunTime: \n{str(parent_1.fitness)}\nSolver Information: \n{parent_1.solver_info}\n\n"
                        f"Memory Library: \n{self.memory_library}"
                    ) if random.random() < self.mem_rate else (
                        f"Problem Description: \n{self.problem_description}\n\n"
                        f"Current Model: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                        f"RunTime: \n{str(parent_1.fitness)}\nSolver Information: \n{parent_1.solver_info}\n\n"
                    )
                    # diagnosis input
                    input_diagnosis_ls.append([
                        {"role": "system",
                        "content": self.system_diagnosis_mutation},
                        {"role": "user",
                        "content": user_diagnosis_mutation}
                    ])
                    parents_indv_ls.append(("mutation", parent_1))

            # generate diagnosis output
            diagnosis_outputs_ls = self.llm_caller.call_batch(input_diagnosis_ls)

            # generate new prompts
            input_evolution_ls = []
            assert len(parents_indv_ls) == len(diagnosis_outputs_ls), "Number of parents and diagnosis outputs do not match!"
            for i, parents in enumerate(parents_indv_ls):
                if parents[0] == "crossover":
                    parent_1, parent_2 = parents[1], parents[2]
                    # crossover system prompt
                    system_crossover = random.choice(self.system_crossover)
                    # crossover user prompt
                    user_crossover = (
                        f"Problem Description: \n{self.problem_description}\n\n"
                        f"Python Template: \n{self.temp_func}\n\n"
                        f"Parent 1: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                        f"RunTime: \n{str(parent_1.fitness)}\n\n"
                        f"Parent 2: \nCode: \n{parent_2.code}\nIdea: \n{parent_2.idea}\n"
                        f"RunTime: \n{str(parent_2.fitness)}\n\n"
                        f"Diagnostic Report: \n{diagnosis_outputs_ls[i]}"
                    )

                    # crossover input 
                    input_evolution_ls.append([
                        {"role": "system",
                        "content": system_crossover},
                        {"role": "user",
                        "content": user_crossover}
                    ])
                else:
                    parent_1 = parents[1]
                    # mutation system prompt
                    system_mutation = random.choice(self.system_mutation)
                    # mutation user prompt
                    user_mutation = (
                        f"Problem Description: \n{self.problem_description}\n\n"
                        f"Python Template: \n{self.temp_func}\n\n"
                        f"Current Model: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                        f"RunTime: \n{str(parent_1.fitness)}\n\n"
                        f"Diagnostic Report: \n{diagnosis_outputs_ls[i]}"
                    )
                    
                    # mutation input
                    input_evolution_ls.append([
                        {"role": "system",
                        "content": system_mutation},
                        {"role": "user",
                        "content": user_mutation}
                    ])

            # generate new individuals in parallel for evolution
            self.new_pop = Population(individuals=[])
            new_outputs = self.llm_caller.call_batch(input_repair_ls + input_evolution_ls)

            # write response to file and add to new_pop
            new_meta_ls = []  # keep parent/diagnosis metadata aligned with successfully parsed new individuals
            num_repair = len(input_repair_ls)
            for idx, new_output in enumerate(new_outputs):
                new_indv = self.write_new_pop(new_output=new_output, new_pop=self.new_pop)
                if new_indv is not None:
                    if idx < num_repair:
                        # repair output: use the original invalid individual as parent
                        new_meta_ls.append((("repair", invalid_pop.individuals[idx]), ""))
                    else:
                        # evolution output: use corresponding parent/diagnosis info
                        evo_idx = idx - num_repair
                        new_meta_ls.append((parents_indv_ls[evo_idx], diagnosis_outputs_ls[evo_idx]))
                self.resp_id += 1  # update the response id
            
            # Collective evaluation of the new population
            if len(self.new_pop) > 0:
                self.eval_population(pop=self.new_pop)

            # Update the valid population while gathering reflection input
            invalid_pop = Population(individuals=[])
            assert len(new_meta_ls) == len(self.new_pop.individuals), "Number of metadata and new individuals do not match!"
            for i, indv in enumerate(self.new_pop.individuals):
                if indv.exec_success == True and indv.fitness is not None:
                    self.valid_pop.add(indv=indv)
                    # add reflection input
                    parents, diagnosis_output = new_meta_ls[i]
                    if parents[0] == "crossover":
                        parent_1, parent_2 = parents[1], parents[2]
                        user_reflection = (
                            f"Problem Description: \n{self.problem_description}\n\n"
                            f"Parent 1: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                            f"RunTime: \n{str(parent_1.fitness)}\nSolver Information: \n{parent_1.solver_info}\n\n"
                            f"Parent 2: \nCode: \n{parent_2.code}\nIdea: \n{parent_2.idea}\n"
                            f"RunTime: \n{str(parent_2.fitness)}\nSolver Information: \n{parent_2.solver_info}\n\n"
                            f"Diagnostic Report: \n{diagnosis_output}\n\n"
                            f"Offspring Model: \nCode: \n{indv.code}\nIdea: \n{indv.idea}\n"
                            f"RunTime: \n{str(indv.fitness)}\nSolver Information: \n{indv.solver_info}"
                        )
                    else:
                        parent_1 = parents[1]
                        user_reflection = (
                            f"Problem Description: \n{self.problem_description}\n\n"
                            f"Parent Model: \nCode: \n{parent_1.code}\nIdea: \n{parent_1.idea}\n"
                            f"RunTime: \n{str(parent_1.fitness)}\nSolver Information: \n{parent_1.solver_info}\n\n"
                            f"Diagnostic Report: \n{diagnosis_output}\n\n"
                            f"Offspring Model: \nCode: \n{indv.code}\nIdea: \n{indv.idea}\n"
                            f"RunTime: \n{str(indv.fitness)}\nSolver Information: \n{indv.solver_info}"
                        )
                    # add reflection input
                    input_reflection_ls.append([
                        {"role": "system",
                        "content": self.system_reflection},
                        {"role": "user",
                        "content": user_reflection}
                    ])
                else:
                    if i >= num_invalid and indv.code is not None:
                        invalid_pop.add(indv=indv)  # record invalid individuals
        
        # Select the top pop_size individuals
        self.valid_pop = self.valid_pop.select_top(pop_size=self.pop_size)
        best_indv = self.valid_pop.individuals[0]
        logging.info(f"Best run: Iter {best_indv.iter_id}, Code {best_indv.resp_id}, Fitness: {best_indv.fitness:.4f}")

        # Call reflection model
        logging.info(f"Iteration {self.iter_id} memory reflection started!")
        refl_outputs = self.llm_caller.call_batch(input_reflection_ls)

        # Memory curation
        refl_items = "\n\n".join(
            f"{item}" for item in refl_outputs
        )
        user_curation = (
            f"Current Memory Library: \n{self.memory_library}\n\n"
            f"New Memory Items: \n{refl_items}"
        )
        input_curation_ls = [
            {"role": "system",
            "content": self.system_curation},
            {"role": "user",
            "content": user_curation}
        ]

        # Call curation model
        logging.info(f"Iteration {self.iter_id} memory curation started!")
        curation_output = self.llm_caller.call(input_curation_ls)
        self.memory_library = curation_output

        # write memory library to final_memory.txt
        with open(os.path.join(self.out_dir, "final_memory.txt"), 'w', encoding="utf-8") as file:
            file.writelines(self.memory_library + '\n')
        
        logging.info(f"Evolution iteration {self.iter_id} finished!\n")

        # Update iteration info
        self.iter_id += 1  # update the iteration id
        self.resp_id = 0  # reset the response id


    def final_evaluation(self):

        logging.info(f"Evolution finished! Total iterations: {self.iter_id - 1}")
        best_indv = self.valid_pop.individuals[0]
        logging.info(f"Training best run: Iter {best_indv.iter_id}, Code {best_indv.resp_id}, Fitness: {best_indv.fitness:.4f}\n")

        logging.info(f"Final evaluation started!")
        # add seed function as baseline for validation
        if not any(indv.iter_id == 0 and indv.resp_id == 0 for indv in self.valid_pop.individuals):
            self.valid_pop.add(indv=self.seed_indv)
        # reset fitness in population
        for indv in self.valid_pop.individuals:
            indv.exec_success = None
            indv.fitness = None
        # sort the valid_pop by iter_id and resp_id
        self.valid_pop.individuals.sort(key=lambda indv: (indv.iter_id, indv.resp_id))

        # revise stdout path
        for indv in self.valid_pop.individuals:
            indv.stdout_path = "final_" + indv.stdout_path

        # final evaluation with validation mode
        self.eval_population(pop=self.valid_pop, mode="validate")

        # filter individuals with valid fitness values
        valid_individuals = [indv for indv in self.valid_pop.individuals if indv.fitness is not None]
        
        if not valid_individuals:
            logging.error("No valid individuals found with fitness values")
            return
        
        # select the best individual with validation fitness value
        best_indv = min(valid_individuals, key=lambda indv: indv.fitness)
        logging.info(f"Validation best run: Iter {best_indv.iter_id}, Code {best_indv.resp_id}, Fitness: {best_indv.fitness:.4f}")

        # write code to final_code.txt
        with open(os.path.join(self.out_dir, "final_code.txt"), 'w', encoding="utf-8") as file:
            file.writelines(best_indv.code + '\n')

        # write memory library to final_memory.txt
        with open(os.path.join(self.out_dir, "final_memory.txt"), 'w', encoding="utf-8") as file:
            file.writelines(self.memory_library + '\n')

        logging.info(f"Final evaluation completed!")

    def eval_population(self, pop, mode="train"):
        
        inner_runs, cores_runs = [], []
        # Evaluate each individual in the population
        for indv in pop.individuals:
            if self.cpu_ls is not None:  # run the individual in parallel with specified core
                while True:  # find available core for the individual run
                    # check current run status for occupied cores (corresponding run not finished yet)
                    occupied_cores = [core for i, core in enumerate(cores_runs) if inner_runs[i].poll() is None]
                    available_cores = [core for core in self.cpu_ls if core not in occupied_cores]
                    if len(available_cores) > 0:  # found available core
                        break
                    else:  # no available core found
                        time.sleep(1)  # wait for 1 second before checking again
                # get the core for the individual
                core = available_cores.pop()  # get the core for the individual
                process = self.eval_individual(indv=indv, core=core, mode=mode)
                inner_runs.append(process)
                cores_runs.append(core)
            else:  # run the individual sequentially without core restriction
                process = self.eval_individual(indv=indv, core=None, mode=mode)
                inner_runs.append(process)
        
        def wait_for_process(indv, process):
            try:
                if process.returncode is None:
                    process.communicate(timeout=self.timeout)
                    actual_returncode = process.poll()
                    if actual_returncode is None:
                        return (indv, process, False, None)
                    elif actual_returncode == 0:
                        return (indv, process, False, None)
                    else:
                        return (indv, process, True, actual_returncode)
                elif process.returncode == 0:
                    return (indv, process, False, None)
                else:
                    return (indv, process, True, process.returncode)
            except subprocess.TimeoutExpired as e:
                logging.info(f"Iteration {indv.iter_id}, Code run {indv.resp_id} execution timeout!")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
                return (indv, process, True, None)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(inner_runs)) as executor:
            futures = [executor.submit(wait_for_process, indv, process) 
                      for indv, process in zip(pop.individuals, inner_runs)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        for indv, process, timed_out, returncode in results:
            if timed_out or returncode is not None and returncode != 0:
                indv.exec_success = False
        
        # Check each individual stdout file and get fitness
        for indv in pop.individuals:
            if indv.exec_success == False:  # no need to check fitness for individuals that failed
                logging.info(f"Iteration {indv.iter_id}, Code run {indv.resp_id} evaluation failed")
                continue
            with open(os.path.join(self.out_dir, indv.stdout_path), 'r',encoding="utf-8") as f:  # read the stdout file
                stdout_str = f.read()
            if filter_traceback(stdout_str) == '':  # no traceback
                try:
                    indv.fitness = float(stdout_str.split('\n')[-2])  # get the fitness from the -2 line of stdout
                    indv.solver_info = json.loads(stdout_str.split('\n')[-4])  # get the solver info from the -4 line of stdout
                    indv.exec_success = True
                    logging.info(f"Iteration {indv.iter_id}, Code run {indv.resp_id}, Fitness: {indv.fitness:.4f}")
                except:
                    logging.info(f"Iteration {indv.iter_id}, Code run {indv.resp_id} invalid stdout format")
            else:
                logging.info(f"Iteration {indv.iter_id}, Code run {indv.resp_id} evaluation failed")

        return pop            

    def eval_individual(self, indv, core=None, mode="train"):

        # write code to code_path
        with open(os.path.join(self.out_dir, indv.code_path), 'w', encoding="utf-8") as file:
            file.writelines(indv.code + '\n')

        # write code to run_file
        with open(self.run_file, 'w', encoding="utf-8") as file:
            file.writelines(indv.code + '\n')

        # execute the python file with flags
        stdout_path = os.path.join(self.out_dir, indv.stdout_path)
        if core is not None:
            with open(stdout_path, 'w', encoding="utf-8") as f:
                process = subprocess.Popen(['taskset', '-c', str(core), 'python', '-u', self.eval_file, stdout_path, mode], stdout=f, stderr=f, preexec_fn=os.setsid)
            # wait for the process to start 
            block_until_running(stdout_path, log_status=True, iter_id=indv.iter_id, resp_id=indv.resp_id, core=core)
        else:
            with open(stdout_path, 'w', encoding="utf-8") as f:
                process = subprocess.Popen(['python', '-u', self.eval_file, stdout_path, mode], stdout=f, stderr=f, preexec_fn=os.setsid)
            # wait for the process to start
            block_until_running(stdout_path, log_status=True, iter_id=indv.iter_id, resp_id=indv.resp_id)
            # wait for the process to complete
            block_until_ending(process, stdout_path, log_status=True, iter_id=indv.iter_id, resp_id=indv.resp_id, timeout=self.timeout)

        return process

    def write_new_pop(self, new_output, new_pop):
        # write response to file
        with open(os.path.join(self.out_dir, f"iter{self.iter_id}_resp{self.resp_id}.txt"), 'w', encoding="utf-8") as file:
            file.writelines(new_output + '\n')
        new_output = extract_json(new_output)  # str -> dict
        if new_output is None or "idea" not in new_output or "code" not in new_output:
            logging.warning(f"Iteration {self.iter_id}, Code gen {self.resp_id} extract json failed")
            return None
        new_idea, new_code = new_output["idea"], extract_code(new_output["code"])  # dict -> str, code
        new_indv = Individual(
            iter_id=self.iter_id,
            resp_id=self.resp_id,
            code=new_code,
            code_path=f"iter{self.iter_id}_code{self.resp_id}.txt",
            stdout_path=f"iter{self.iter_id}_stdout{self.resp_id}.txt",
            idea=new_idea,
            exec_success=None,  # no evaluated yet
            fitness=None,  # not evaluated yet
            solver_info=None,  # not evaluated yet
        )
        logging.info(f"Iteration {self.iter_id}, Code gen {self.resp_id} successful")
        new_pop.add(indv=new_indv)  # add the new individual to the new population (no evaluated yet)
        return new_indv

    def save_population(self, filepath="pop_ckpt.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump(self.valid_pop, f)
        logging.info(f"Population saved to {filepath}")

    def load_population(self, filepath="pop_ckpt.pkl"):
        with open(filepath, 'rb') as f:
            self.valid_pop = pickle.load(f)
        logging.info(f"Population loaded from {filepath}")
