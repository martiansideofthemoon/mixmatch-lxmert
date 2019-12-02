import itertools
import collections
import glob
import getpass
import os
import datetime
import subprocess
import string
import sys


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def get_run_id():
    filename = "logs/expts_nlvr2.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts) / 5
    return run_id

top_details = "Tuning hyperparameters for mixmatch setup"

hyperparameters = [
    [('batch_size',), [32]],
    [('extra_flags',), ['--train_data_fraction 0.1 --batch_self_train', '--train_data_fraction 0.2 --batch_self_train', '--train_data_fraction 0.5 --batch_self_train']],
]


run_id = int(get_run_id())
key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []

username = getpass.getuser()

for combo in combinations:
    # Write the scheduler scripts
    with open("nlvr2_finetune_template_%s.sh" % username, 'r') as f:
        schedule_script = f.read()
    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    od = collections.OrderedDict(sorted(combo.items()))
    lower_details = ""
    for k, v in od.items():
        lower_details += "%s = %s, " % (k, str(v))
    # removing last comma and space
    lower_details = lower_details[:-2]

    combo["top_details"] = top_details
    combo["lower_details"] = lower_details
    combo["job_id"] = run_id
    print("Scheduling Job #%d" % run_id)

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))

    schedule_script += "\n"
    # Write schedule script
    script_name = 'schedulers/schedule_nlvr2_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)
    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name + " by " + username + "\n" + \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
        top_details + "\n" + \
        lower_details + "\n\n"
    with open("logs/expts_nlvr2.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1

# schedule jobs
for script in scripts:
    command = "sbatch %s" % script
    print(subprocess.check_output(command, shell=True))
