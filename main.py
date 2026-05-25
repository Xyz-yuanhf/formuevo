import json
import logging

from evolution import FormuEvo

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    
    # load config
    with open("configs/run_cfg.json", 'r') as f:
        config = json.load(f)

    # run evolution
    formu_evo = FormuEvo(config=config)
    formu_evo.read_prompt()
    formu_evo.seed_evaluation()
    formu_evo.init_population()
    for _ in range(formu_evo.max_iter):
        formu_evo.step_evolution()
    formu_evo.final_evaluation()
    