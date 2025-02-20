# -*- coding: future_fstrings -*-
import sys, os, time

t0 = time.time()
import socket
import numpy as np
import torch
from ruamel.yaml import YAML
from absl import flags
from utils import system, logger
from pathlib import Path
import psutil

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import omegaconf

# FLAGS = flags.FLAGS
# flags.DEFINE_string("cfg", None, "path to configuration file")
# flags.DEFINE_string("env", None, "env_name")
# flags.DEFINE_string("algo", None, '["td3", "sac", "sacd"]')
#
# flags.DEFINE_boolean("automatic_entropy_tuning", None, "for [sac, sacd]")
# flags.DEFINE_float("target_entropy", None, "for [sac, sacd]")
# flags.DEFINE_float("entropy_alpha", None, "for [sac, sacd]")
#
# flags.DEFINE_integer("seed", None, "seed")
# flags.DEFINE_integer("cuda", None, "cuda device id")
# flags.DEFINE_boolean("debug", False, "debug mode")
#
# flags.DEFINE_string("chkpt_path", None, "checkpoint to load")
# flags.DEFINE_boolean("render", False, "should enjoy be rendered")
# flags.DEFINE_integer("num_runs", 100, "number of eavlauation runs")
#
# flags.FLAGS(sys.argv)
yaml = YAML()
# v = yaml.load(open(FLAGS.cfg))

# overwrite config params
# if FLAGS.env is not None:
#     v["env"]["env_name"] = FLAGS.env
# if FLAGS.algo is not None:
#     v["policy"]["algo_name"] = FLAGS.algo
#
# seq_model, algo = v["policy"]["seq_model"], v["policy"]["algo_name"]
# assert seq_model in ["mlp", "lstm", "gru", "lstm-mlp", "gru-mlp"]
# assert algo in ["td3", "sac", "sacd"]
#
# if FLAGS.automatic_entropy_tuning is not None:
#     v["policy"][algo]["automatic_entropy_tuning"] = FLAGS.automatic_entropy_tuning
# if FLAGS.entropy_alpha is not None:
#     v["policy"][algo]["entropy_alpha"] = FLAGS.entropy_alpha
# if FLAGS.target_entropy is not None:
#     v["policy"][algo]["target_entropy"] = FLAGS.target_entropy
#
# if FLAGS.cuda is not None:
#     v["cuda"] = FLAGS.cuda


@hydra.main(version_base="1.1", config_path="..", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    # USE SAME SEED FOR ALL EVALUATION RUNS
    seed = 21
    # seed = v["seed"]
    system.reproduce(seed)

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    pid = str(os.getpid())
    if "SLURM_JOB_ID" in os.environ:
        pid += "_" + str(os.environ["SLURM_JOB_ID"])  # use job id

    # set gpu
    set_gpu_mode(torch.cuda.is_available() and cfg.cuda >= 0, cfg.cuda)

    v = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    env_type = v["env"]["env_type"]
    if len(v["env"]["env_name"].split("-")) == 3:
        # pomdp env: name-{F/P/V}-v0
        env_name, pomdp_type, _ = v["env"]["env_name"].split("-")
        env_name = env_name + "/" + pomdp_type
    else:
        env_name = v["env"]["env_name"]

    exp_id = f"{to_absolute_path('logs')}/"
    exp_id += f"{env_type}/{env_name}/"

    seq_model, algo = v["policy"]["seq_model"], v["policy"]["algo_name"]
    assert seq_model in ["mlp", "lstm", "gru", "lstm-mlp", "gru-mlp"]
    assert algo in ["td3", "sac", "sacd"]

    if seq_model == "mlp":
        algo_name = f"Markovian_{algo}"
    else:  # rnn
        if "rnn_num_layers" in v["policy"]:
            rnn_num_layers = v["policy"]["rnn_num_layers"]
            if rnn_num_layers == 1:
                rnn_num_layers = ""
            else:
                rnn_num_layers = str(rnn_num_layers)
        else:
            rnn_num_layers = ""
        exp_id += f"{algo}_{rnn_num_layers}{seq_model}"
        if "separate" in v["policy"] and v["policy"]["separate"] == False:
            exp_id += "_shared"
    exp_id += "/"

    if algo in ["sac", "sacd"]:
        if not v["policy"][algo]["automatic_entropy_tuning"]:
            exp_id += f"alpha-{v['policy'][algo]['entropy_alpha']}/"
        elif "target_entropy" in v["policy"]:
            exp_id += f"ent-{v['policy'][algo]['target_entropy']}/"

    exp_id += f"gamma-{v['policy']['gamma']}/"

    if seq_model != "mlp":
        exp_id += f"len-{v['train']['sampled_seq_len']}/bs-{v['train']['batch_size']}/"
        # exp_id += f"baseline-{v['train']['sample_weight_baseline']}/"
        exp_id += f"freq-{v['train']['num_updates_per_iter']}/"
        # assert v["policy"]["observ_embedding_size"] > 0
        policy_input_str = "o"
        if v["policy"]["action_embedding_size"] > 0:
            policy_input_str += "a"
        if v["policy"]["reward_embedding_size"] > 0:
            policy_input_str += "r"
        exp_id += policy_input_str + "/"

    os.makedirs(exp_id, exist_ok=True)
    log_folder = os.path.join(exp_id, system.now_str())
    logger_formats = ["stdout", "log", "csv"]
    if v["eval"]["log_tensorboard"]:
        logger_formats.append("tensorboard")
    logger.configure(v, dir=log_folder, format_strs=logger_formats, precision=4)
    logger.log(f"preload cost {time.time() - t0:.2f}s")

    os.system(f"cp -r policies/ {log_folder}")
    yaml.dump(v, Path(f"{log_folder}/variant_{pid}.yml"))
    # key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
    # logger.log("\n".join(f.serialize() for f in key_flags) + "\n")
    logger.log("pid", pid, socket.gethostname())
    os.makedirs(os.path.join(logger.get_dir(), "save"))

    # start training
    learner = Learner(
        env_args=v["env"],
        train_args=v["train"],
        eval_args=v["eval"],
        policy_args=v["policy"],
        seed=seed,
    )

    logger.log(
        f"total RAM usage: {psutil.Process().memory_info().rss / 1024 ** 3 :.2f} GB\n"
    )

    print(f"config {cfg}")
    if cfg.chkpt_path is not None:
        learner.enjoy(cfg.chkpt_path, cfg.render, cfg.num_runs)


if __name__ == "__main__":
    main()
