from utils.gfn_diffusion import diffusion_resample
from utils.visualization import *
from utils.simple_io import *
from utils.args import fetch_args

args, state = fetch_args(experiment_run=False)

exp_names = [expn for expn in get_filenames(path=args.save_folder) if '.DS' not in expn]
exp_paths = [f"{args.save_folder}/{expn}" for expn in exp_names]

exp_args = {exp_name: DictObj(load_dict_from_file(f"{exp_path}/run_args.json")) for exp_name, exp_path in zip(exp_names, exp_paths) if file_exists(f"{exp_path}/run_args.json")}
exp_logs = {exp_name: load_dict_from_file(f"{exp_path}/run_logs.json") for exp_name, exp_path in zip(exp_names, exp_paths) if file_exists(f"{exp_path}/run_logs.json")}

# plot_exp_logs(exp_logs, exp_args)
# plot_separate_exp_logs(exp_logs, exp_args)
diffusion_resample(
    exp_args,
    exp_paths=[exp_path for exp_path in exp_paths if file_exists(f"{exp_path}/run_args.json")],
    batch_size=args.plot_batch_size,
    device=args.device
)


