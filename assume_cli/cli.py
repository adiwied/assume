#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import argcomplete
import yaml
from sqlalchemy import make_url
import time as time

import wandb


def db_uri_completer(prefix, parsed_args, **kwargs):
    return {
        "sqlite:///example.db": "example",
        f"sqlite:///examples/local_db/{parsed_args.scenario}.db": "current scenario",
        "sqlite://": "in-memory",
        "postgresql://assume:assume@localhost:5432/assume": "localhost",
        "postgresql://assume:assume@assume_db:5432/assume": "docker",
        "mysql://username:password@localhost:3306/database": "mysql",
    }


def config_directory_completer(prefix, parsed_args, **kwargs):
    directory = Path(parsed_args.input_path)
    if directory.is_dir():
        config_folders = [
            folder
            for folder in directory.iterdir()
            if folder.is_dir() and (folder / "config.yaml").exists()
        ]
        return [
            folder.name for folder in config_folders if folder.name.startswith(prefix)
        ]
    return [""]


def config_case_completer(prefix, parsed_args, **kwargs):
    config_file = (
        Path(parsed_args.input_path) / Path(parsed_args.scenario) / "config.yaml"
    )
    if config_file.is_file():
        with open(str(config_file)) as f:
            config = yaml.safe_load(f)
        return list(config.keys())
    return [""]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command Line Interface for ASSUME simulations",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        help="name of the scenario file which should be used",
        default="example_01a",
        type=str,
    ).completer = config_directory_completer
    parser.add_argument(
        "-c",
        "--case-study",
        help="name of the case in that scenario which should be simulated",
        default="",
        type=str,
    ).completer = config_case_completer
    parser.add_argument(
        "-csv",
        "--csv-export-path",
        help="optional path to the csv export",
        default="",
        type=str,
    ).completer = argcomplete.DirectoriesCompleter()
    parser.add_argument(
        "-db",
        "--db-uri",
        help="uri string for a database",
        default="",
        type=str,
    ).completer = db_uri_completer
    parser.add_argument(
        "-i",
        "--input-path",
        help="path to the input folder",
        default="examples/inputs",
        type=str,
    ).completer = argcomplete.DirectoriesCompleter()
    parser.add_argument(
        "-l",
        "--loglevel",
        help="logging level used for file log",
        default="INFO",
        type=str,
        metavar="LOGLEVEL",
        choices=set(logging._nameToLevel.keys()),
    )

    parser.add_argument(
        "-p",
        "--parallel",
        help="run simulation with multiple processes",
        action="store_true",
    )
    return parser


def cli(args=None):
    if not args:
        args = sys.argv[1:]
    parser = create_parser()

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args)
    name = args.scenario
    if args.db_uri:
        db_uri = make_url(args.db_uri)
    else:
        db_uri = f"sqlite:///./examples/local_db/{name}.db"

    # add these two weird hacks for now
    warnings.filterwarnings("ignore", "coroutine.*?was never awaited.*")
    logging.getLogger("asyncio").setLevel("FATAL")

    try:
        # import package after argcomplete.autocomplete
        # to improve autocompletion speed
        from assume import World
        from assume.scenario.loader_csv import load_scenario_folder, run_learning

        os.makedirs("./examples/local_db", exist_ok=True)

        if args.parallel:
            distributed_role = True
            addr = ("localhost", 9100)
        else:
            distributed_role = None
            addr = "world"

        world = World(
            database_uri=db_uri,
            export_csv_path=args.csv_export_path,
            log_level=args.loglevel,
            distributed_role=distributed_role,
            addr=addr,
        )
        load_scenario_folder(
            world,
            inputs_path=args.input_path,
            scenario=args.scenario,
            study_case=args.case_study,
        )
        # set up the wandb run
        _setup_wandb(world.learning_config)
        

        if world.learning_config.get("learning_mode", False):
            run_learning(
                world,
                inputs_path=args.input_path,
                scenario=args.scenario,
                study_case=args.case_study,
            )

        world.run()

    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Simulation aborted")


def _setup_wandb(learning_config):
        if wandb.run is not None:
            return
        
        try:
            # Try to read API key from file
            try:
                with open('wandb_key.txt', 'r') as f:
                    wandb.login(key=f.read().strip(), relogin=True)
            except FileNotFoundError:
                print("Warning: wandb_key.txt not found. Attempting to use existing credentials.")
            except Exception as e:
                print(f"Warning: Failed to read wandb API key: {e}")

            run_config = {
                "algorithm": {
                    "name": learning_config["algorithm"],
                    "learning_rate": learning_config["learning_rate"],
                    "batch_size": learning_config["ppo"]["batch_size"],
                    "gamma": learning_config["ppo"]["gamma"],
                    "gradient_steps": learning_config["gradient_steps"]
                },
                "architecture": {
                    "actor": learning_config["ppo"]["actor_architecture"],
                    "device": str(learning_config["device"])
                },
                "training": {
                    "total_episodes": learning_config["training_episodes"],
                    "train_freq": learning_config["ppo"]["train_freq"],
                    #": learning_config["episodes_collecting_initial_experience"]
                }
            }

            run_config["algorithm"].update({
                "clip_ratio": learning_config["ppo"]["clip_ratio"],
                "entropy_coef": learning_config["ppo"]["entropy_coef"],
                "value_coeff": learning_config["ppo"]["vf_coef"],
                "max_grad_norm": learning_config["ppo"]["max_grad_norm"],
                "gae_lambda": learning_config["ppo"]["gae_lambda"]
            })

            run_name = f"ppo_{learning_config["ppo"]["batch_size"]}_{learning_config["learning_rate"]}_{time.time()}"
            
            if learning_config["perform_evaluation"]:
                run_name += "_eval"
            
            wandb.init(
                project="ASSUME-PPO",
                name=run_name, 
                config=run_config,
                group=learning_config.get("experiment_group", None),  # Group related runs
                tags=[
                    "ppo",
                    f"batch_{learning_config["ppo"]["batch_size"]}",
                    "evaluation" if learning_config["perform_evaluation"] else "training",
                    # learning_config.get("custom_tag", None)
                ],
                mode="offline" if learning_config.get("wandb_offline", False) else "online",
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True  # Disable system stats collection
                    , 
                ),
            )
            # Log code-saving configuration
            wandb.run.log_code = True  # Save source code
                        
        except Exception as e:
            print(f"Failed to initialize wandb: {e}") 
            print("Continuing without logging")


if __name__ == "__main__":
    cli()

    # args = "-s example_03 -db postgresql://assume:assume@localhost:5432/assume"

    # cli(args.split(" "))
