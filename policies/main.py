# -*- coding: future_fstrings -*-
import sys, os, time
import glob

t0 = time.time()
import socket
import numpy as np
import torch

# from ruamel.yaml import YAML
# from absl import flags
from utils import system, logger
from pathlib import Path
import psutil

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import omegaconf
import wandb

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
# flags.FLAGS(sys.argv)
# yaml = YAML()
#


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    # v = yaml.load(open(FLAGS.cfg))

    # overwrite config params
    # if FLAGS.env is not None:
    #     v["env"]["env_name"] = FLAGS.env
    # if FLAGS.algo is not None:
    #     v["policy"]["algo_name"] = FLAGS.algo

    print(cfg)
    seq_model, algo = cfg.policy.seq_model, cfg.policy.algo_name
    assert seq_model in ["mlp", "lstm", "gru", "lstm-mlp", "gru-mlp"]
    assert algo in ["td3", "sac", "sacd"]

    # if FLAGS.automatic_entropy_tuning is not None:
    #     v["policy"][algo]["automatic_entropy_tuning"] = FLAGS.automatic_entropy_tuning
    # if FLAGS.entropy_alpha is not None:
    #     v["policy"][algo]["entropy_alpha"] = FLAGS.entropy_alpha
    # if FLAGS.target_entropy is not None:
    #     v["policy"][algo]["target_entropy"] = FLAGS.target_entropy

    # if FLAGS.seed is not None:
    #     v["seed"] = FLAGS.seed
    # if FLAGS.cuda is not None:
    #     v["cuda"] = FLAGS.cuda

    # system: device, threads, seed, pid
    seed = cfg.seed
    system.reproduce(seed)

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    pid = str(os.getpid())
    # if "SLURM_JOB_ID" in os.environ:
    #     pid += "_" + str(os.environ["SLURM_JOB_ID"])  # use job id

    # set gpu
    set_gpu_mode(torch.cuda.is_available() and cfg.cuda >= 0, cfg.cuda)

    # logs
    # if FLAGS.debug:
    #     exp_id = "debug/"
    # else:
    exp_id = f"{to_absolute_path('logs')}/"

    env_type = cfg.env.env_type
    if len(cfg.env.env_name.split("-")) == 3:
        # pomdp env: name-{F/P/V}-v0
        env_name, pomdp_type, _ = cfg.env.env_name.split("-")
        env_name = env_name + "/" + pomdp_type
    else:
        env_name = cfg.env.env_name
    exp_id += f"{env_type}/{env_name}/"

    if seq_model == "mlp":
        algo_name = f"Markovian_{algo}"
    else:  # rnn
        if "rnn_num_layers" in cfg.policy:
            rnn_num_layers = cfg.policy.rnn_num_layers
            if rnn_num_layers == 1:
                rnn_num_layers = ""
            else:
                rnn_num_layers = str(rnn_num_layers)
        else:
            rnn_num_layers = ""
        exp_id += f"{algo}_{rnn_num_layers}{seq_model}"
        if "separate" in cfg.policy and cfg.policy.separate == False:
            exp_id += "_shared"
    exp_id += "/"

    v = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if algo in ["sac", "sacd"]:
        if not v["policy"][algo]["automatic_entropy_tuning"]:
            exp_id += f"alpha-{v['policy'][algo]['entropy_alpha']}/"
        elif "target_entropy" in v["policy"]:
            exp_id += f"ent-{v['policy'][algo]['target_entropy']}/"

    exp_id += f"gamma-{cfg.policy.gamma}/"

    if seq_model != "mlp":
        exp_id += f"len-{cfg.train.sampled_seq_len}/bs-{cfg.train.batch_size}/"
        # exp_id += f"baseline-{cfg.train.sample_weight_baseline}/"
        exp_id += f"freq-{cfg.train.num_updates_per_iter}/"
        # assert v["policy"]["observ_embedding_size"] > 0
        policy_input_str = "o"
        if cfg.policy.action_embedding_size > 0:
            policy_input_str += "a"
        if cfg.policy.reward_embedding_size > 0:
            policy_input_str += "r"
        exp_id += policy_input_str + "/"

    os.makedirs(exp_id, exist_ok=True)
    log_folder = os.path.join(exp_id, system.now_str())
    logger_formats = ["stdout", "log", "csv", "wandb"]
    if cfg.eval.log_tensorboard:
        logger_formats.append("tensorboard")
    logger.configure(v, dir=log_folder, format_strs=logger_formats, precision=4)
    logger.log(f"preload cost {time.time() - t0:.2f}s")

    os.system(f"cp -r {to_absolute_path('policies')}/ {log_folder}")

    if "merge" in cfg.env.env_name:
        os.system(f"cp -r {to_absolute_path('highway-env')}/ {log_folder}")

        print(f"Working directory : {os.getcwd()}")
        artifact = wandb.run.log_code(
            f"{to_absolute_path('highway-env')}",
            name="Simulation_Code",
            include_fn=lambda path: path.endswith(".py")
            or path.endswith(".pyx")
            or path.endswith("c_utils.c"),
        )
        wandb.run.log_artifact(artifact)
        wandb.run.use_artifact(artifact, type="code")
        artifact.wait()

        artifact = wandb.run.log_code(
            f"{to_absolute_path('policies')}",
            name="Training_Code",
        )
        wandb.run.log_artifact(artifact)
        wandb.run.use_artifact(artifact, type="code")
        artifact.wait()

    # yaml.dump(v, Path(f"{log_folder}/variant_{pid}.yml"))
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

    learner.train()


if __name__ == "__main__":
    main()
