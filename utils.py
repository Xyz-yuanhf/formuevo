import re
import os
import json
import signal
import logging
import subprocess

# Extract python code and idea from (LLM generated) output
def extract_json(output):
    # try to load json data
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        try:
            # remove markdown wrap if exists
            output = output.strip()
            if output.startswith("```json"):
                output = output[7:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
            return json.loads(output)
        except json.JSONDecodeError:
            # manually extract json data by \{ and \}
            lines = output.split('\n')
            start, end = None, None
            for i, line in enumerate(lines):
                if '{' in line and start is None:
                    start = i
                if '}' in line:
                    end = i
            if start is not None and end is not None:
                # remove any leading / trailing whitespace
                json_str = '\n'.join(lines[start:end+1]).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
            else:
                return None

# Extract executable code from (LLM generated) python code
def extract_code(content):
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # find the line that starts with "def" and the line that starts with "return"
        #  extract the code between them
        lines = content.split('\n')
        start, end = None, None
        for i, line in enumerate(lines):
            if 'import' in line and start is None:
                start = i
            if 'return' in line:
                end = i
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None:
        return None
    
    # add import statements if not present
    if "GRB" in code_string and "from gurobipy import GRB" not in code_string:  # GRB
        code_string = "from gurobipy import GRB\n" + code_string
    if "gp" in code_string and "import gurobipy as gp" not in code_string:  # gurobipy
        code_string = "import gurobipy as gp\n" + code_string
    if "np" in code_string and "import numpy as np" not in code_string:  # numpy
        code_string = "import numpy as np\n" + code_string
    if "scipy" in code_string and "import scipy" not in code_string:  # scipy
        code_string = "import scipy\n" + code_string
    if "sp" in code_string and "import scipy as sp" not in code_string:  # scipy
        code_string = "import scipy as sp\n" + code_string
    if "math" in code_string and "import math" not in code_string:  # math
        code_string = "import math\n" + code_string
    if "collections" in code_string and "import collections" not in code_string:  # collections
        code_string = "import collections\n" + code_string
    if "Counter" in code_string and "from collections import Counter" not in code_string:  # Counter
        code_string = "from collections import Counter\n" + code_string
    return code_string

# Block until the evaluation is finished
def block_until_running(stdout_path, log_status=False, iter_id=-1, resp_id=-1, core=None):
    # Ensure that the evaluation has started before moving on
    while True:
        with open(stdout_path, 'r', encoding="utf-8") as file:
            log = file.read()
        if len(log) > 0:
            if log_status and "Traceback" in log:
                if core is not None:
                    logging.warning(f"Iteration {iter_id}, Code run {resp_id} execution error on core {core}! (see {print_hyperlink(stdout_path, 'stdout')})")
                else:
                    logging.warning(f"Iteration {iter_id}, Code run {resp_id} execution error! (see {print_hyperlink(stdout_path, 'stdout')})")
            else:
                if core is not None:
                    logging.info(f"Iteration {iter_id}, Code run {resp_id} started on core {core}! (see {print_hyperlink(stdout_path, 'stdout')})")
                else:
                    logging.info(f"Iteration {iter_id}, Code run {resp_id} started! (see {print_hyperlink(stdout_path, 'stdout')})")
            break

# Block until the process is finished
def block_until_ending(process, stdout_path=None, log_status=False, iter_id=-1, resp_id=-1, timeout=None):
    # wait for the process to complete
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        logging.warning(f"Iteration {iter_id}, Code run {resp_id} execution timeout!")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=5)
        return
    
    with open(stdout_path, 'r', encoding="utf-8") as file:
            log = file.read()
    if log_status:
        if "Traceback" in log:
            logging.warning(f"Iteration {iter_id}, Code run {resp_id} execution error! (see {print_hyperlink(stdout_path, 'stdout')})")
        else:
            logging.info(f"Iteration {iter_id}, Code run {resp_id} completed! (see {print_hyperlink(stdout_path, 'stdout')})")

# Filter traceback from stdout
def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # return an empty string if no Traceback is found

# Print hyperlink to file or folder for convenient navigation
def print_hyperlink(path, text=None):
    # Format: \033]8;;file:///path/to/file\033\\text\033]8;;\033\\
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"

# Filter physical cpu cores for Linux
def get_physical_cores(cpu_ls):
    selected_physical_cores = []
    covered_siblings = set()  # record the cores that have been processed as siblings of a physical core already

    for cpu_id in cpu_ls:
        # if this core has been processed as a sibling of a physical core already, skip it
        if cpu_id in covered_siblings:
            continue
        
        # read the thread siblings list from Linux kernel provides
        topology_path = f"/sys/devices/system/cpu/cpu{cpu_id}/topology/thread_siblings_list"
        
        if os.path.exists(topology_path):
            try:
                with open(topology_path, 'r') as f:
                    siblings_str = f.read().strip()
                
                # siblings_str reformat
                siblings = set()
                for part in siblings_str.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        siblings.update(range(start, end + 1))
                    else:
                        siblings.add(int(part))
                
                # Add all parsed sibling cores to the blacklist (skip in subsequent loops)
                covered_siblings.update(siblings)
                selected_physical_cores.append(cpu_id)
                
            except Exception as e:
                logging.info(f"Error reading CPU {cpu_id} topology: {e}, treating as independent core.")
                selected_physical_cores.append(cpu_id)
        else:
            # If topology file is not found (extremely rare), treat it as an independent physical core
            selected_physical_cores.append(cpu_id)
    
    selected_physical_cores.sort()
    return selected_physical_cores