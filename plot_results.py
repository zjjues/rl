import argparse
from collections import defaultdict
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ALPHA = 0.2
THRESHOLD_FOR_NUM_ALGS_UNTIL_LEGEND_BELOW_PLOT = 6
THRESHOLD_FOR_ALG_NAME_LENGTH_UNTIL_LEGEND_BELOW_PLOT = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage7-json",
        type=str,
        default=None,
        help="Path to experiments/stage7_results.json for Stage 7 PNG report plots.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        help="Path to directory containing (multiple) results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_return_mean",
        help="Metric to plot",
    )
    parser.add_argument(
        "--filter_by_algs",
        nargs="+",
        default=[],
        help="Filter results by algorithm names. Only showing results for algorithms that contain any of the specified strings in their names.",
    )
    parser.add_argument(
        "--filter_by_envs",
        nargs="+",
        default=[],
        help="Filter results by environment names. Only showing results for environments that contain any of the specified strings in their names.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=Path.cwd(),
        help="Path to directory to save plots to",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=None,
        help="Minimum value for y-axis",
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=None,
        help="Maximum value for y-axis",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use log scale for y-axis",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=None,
        help="Smoothing window for data",
    )
    parser.add_argument(
        "--best_per_alg",
        action="store_true",
        help="Plot only best performing config per alg",
    )
    return parser.parse_args()


def _stage7_final_round(payload):
    rounds = payload.get("rounds", [])
    if not rounds:
        raise ValueError("Stage 7 JSON does not contain any rounds")
    return rounds[-1]


def _stage7_variant_label(summary):
    return summary["variant"]["display_name"]


def _stage7_collect_curve(seed_results, metric):
    curves = []
    for result in seed_results:
        points = [
            (int(float(item["episode"])), float(item[metric]))
            for item in result.get("logs", [])
            if metric in item and "episode" in item
        ]
        if points:
            episodes, values = zip(*points)
            curves.append((np.asarray(episodes), np.asarray(values)))
    if not curves:
        return np.asarray([]), np.asarray([]), np.asarray([])
    common = sorted(set.intersection(*(set(ep.tolist()) for ep, _ in curves)))
    x = np.asarray(common, dtype=np.int32)
    rows = []
    for episodes, values in curves:
        lookup = {int(ep): float(value) for ep, value in zip(episodes, values)}
        rows.append([lookup[int(ep)] for ep in x])
    data = np.asarray(rows, dtype=np.float32)
    return x, data.mean(axis=0), data.std(axis=0)


def plot_stage7_results(stage7_json: Path, save_dir: Path) -> None:
    with open(stage7_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    save_dir.mkdir(parents=True, exist_ok=True)
    round_payload = _stage7_final_round(payload)
    summaries = round_payload["summaries"]
    is_legacy_stage7 = payload.get("stage") == 7
    study_label = "Stage 7" if is_legacy_stage7 else str(payload.get("study_level", "Study")).title()
    file_prefix = "stage7" if is_legacy_stage7 else "study"

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 180, "savefig.dpi": 240})

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for key, summary in summaries.items():
        if summary["variant"]["family"] not in {"baseline", "representation_control"}:
            continue
        seed_results = []
        round_dir = Path(payload.get("work_dir", "experiments/closed_loop_tuning")) / f"round_{round_payload['round']:02d}" / key
        for seed in payload.get("seeds", []):
            result_path = round_dir / f"seed_{seed}" / "result.json"
            if result_path.exists():
                with open(result_path, "r", encoding="utf-8") as f:
                    seed_results.append(json.load(f))
        x, mean, std = _stage7_collect_curve(seed_results, "episode_return")
        if x.size:
            ax.plot(x, mean, linewidth=2.0, label=_stage7_variant_label(summary))
            ax.fill_between(x, mean - std, mean + std, alpha=0.16)
    ax.set_title(f"{study_label} Training Reward Convergence")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / f"{file_prefix}_training_reward_convergence.png")
    plt.close(fig)

    baseline_keys = [
        key
        for key, value in summaries.items()
        if value["variant"]["family"] in {"baseline", "representation_control"}
    ]
    tiers = ["easy", "medium", "hard"]
    x = np.arange(len(tiers), dtype=np.float32)
    width = 0.8 / max(len(baseline_keys), 1)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, key in enumerate(baseline_keys):
        summary = summaries[key]
        means = [summary["risk_tiers"][tier]["collision_rate_mean"] for tier in tiers]
        stds = [summary["risk_tiers"][tier]["collision_rate_std"] for tier in tiers]
        offset = (idx - (len(baseline_keys) - 1) / 2.0) * width
        ax.bar(x + offset, means, yerr=stds, width=width, capsize=3, label=_stage7_variant_label(summary))
    ax.axhline(0.30, color="#8c2d04", linestyle="--", linewidth=1.2)
    ax.set_title(f"{study_label} Evaluation Collision Rate by Risk Tier")
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Collision Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / f"{file_prefix}_risk_tier_collision_rates.png")
    plt.close(fig)

    ablation_keys = [key for key, value in summaries.items() if value["variant"]["family"] == "ablation"]
    labels = [_stage7_variant_label(summaries[key]) for key in ablation_keys]
    hard_collision = [summaries[key]["risk_tiers"]["hard"]["collision_rate_mean"] for key in ablation_keys]
    hard_task = [summaries[key]["risk_tiers"]["hard"]["task_completion_mean"] for key in ablation_keys]
    latency = [summaries[key]["replanning_latency_mean"] for key in ablation_keys]
    x = np.arange(len(ablation_keys), dtype=np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    axes[0].bar(x, hard_collision, color="#4c78a8")
    axes[0].axhline(0.30, color="#8c2d04", linestyle="--", linewidth=1.0)
    axes[0].set_title("Hard Collision")
    axes[0].set_ylabel("Rate")
    axes[1].bar(x, hard_task, color="#59a14f")
    axes[1].axhline(0.75, color="#8c2d04", linestyle="--", linewidth=1.0)
    axes[1].set_title("Hard Task Completion")
    axes[2].bar(x, latency, color="#f28e2b")
    axes[2].axhline(3.5, color="#8c2d04", linestyle="--", linewidth=1.0)
    axes[2].set_title("Re-planning Latency")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.suptitle(f"{study_label} Ablation Performance Comparison")
    fig.tight_layout()
    fig.savefig(save_dir / f"{file_prefix}_ablation_comparison.png")
    plt.close(fig)


def extract_alg_name_from_config(config):
    return config["name"]


def extract_env_name_from_config(config):
    env = config["env"]
    if "map_name" in config["env_args"]:
        env_name = config["env_args"]["map_name"]
    elif "key" in config["env_args"]:
        env_name = config["env_args"]["key"]
    else:
        env_name = None
    return f"{env}_{env_name}"


def load_results(path, metric):
    path = Path(path)
    metrics_files = path.glob("**/metrics.json")

    # map (env_args, env_name, common_reward, reward_scalarisation) -> alg_name -> config-str -> (config, steps, values)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for file in metrics_files:
        # load json
        with open(file, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                warnings.warn(f"Could not load metrics from {file} --> skipping")
                continue

        # find corresponding config file
        config_file = file.parent / "config.json"
        if not config_file.exists():
            warnings.warn(f"No config file found for {file} --> skipping")
            continue
        else:
            with open(config_file, "r") as f:
                config = json.load(f)

        if metric in metrics:
            steps = metrics[metric]["steps"]
            values = metrics[metric]["values"]
        elif "return" in metric and not config["common_reward"]:
            warnings.warn(
                f"Metric {metric} not found in {file}. To plot returns for runs with individual rewards (common_reward=False), you can plot 'total_return' metrics or returns of individual agents --> skipping"
            )
            continue
        else:
            warnings.warn(f"Metric {metric} not found in {file} --> skipping")
            continue
        del config["seed"]

        alg_name = extract_alg_name_from_config(config)
        env_name = extract_env_name_from_config(config)
        env_args = config["env_args"]
        common_reward = config["common_reward"]
        reward_scalarisation = config["reward_scalarisation"]

        data[(str(env_args), env_name, common_reward, reward_scalarisation)][alg_name][
            str(config)
        ].append((config, steps, values))
    return data


def filter_results(data, filter_by_algs, filter_by_envs):
    """
    Filter data to only contain results for algorithms and envs that contain any of the specified strings in their names.
    :param data: dict with results
    :param filter_by_algs: list of strings to filter algorithms by
    :param filter_by_envs: list of strings to filter environments by
    :return: filtered data
    """
    filtered_data = data.copy()

    # filter envs
    if filter_by_envs:
        delete_env_keys = set()
        for key in data:
            env_name = key[1]
            if not any(env in env_name for env in filter_by_envs):
                delete_env_keys.add(key)
        for key in delete_env_keys:
            del filtered_data[key]

    if filter_by_algs:
        for env_key, env_data in filtered_data.items():
            delete_alg_keys = set()
            for alg_name in env_data:
                if not any(alg in alg_name for alg in filter_by_algs):
                    delete_alg_keys.add(alg_name)
            for key in delete_alg_keys:
                del filtered_data[env_key][key]

    return filtered_data


def aggregate_results(data):
    """
    Aggregate results with mean and std over runs of the same config
    :param data: dict mapping key -> list of (config, steps, values)
    :return: aggregated data as dict with key -> (config, steps, means, stds)
    """
    agg_data = defaultdict(list)
    for key, results in data.items():
        config = results[0][0]
        all_steps = []
        all_values = []
        max_len = max([len(steps) for _, steps, _ in results])

        for _, steps, values in results:
            if len(steps) != max_len:
                # append np.nan to values to make sure they have the same length
                steps = np.concatenate([steps, np.full(max_len - len(steps), np.nan)])
                values = np.concatenate(
                    [values, np.full(max_len - len(values), np.nan)]
                )
            all_steps.append(steps)
            all_values.append(values)

        agg_steps = np.nanmean(np.stack(all_steps), axis=0)
        values = np.stack(all_values)
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0)
        agg_data[key] = (config, agg_steps, means, stds)
    return agg_data


def smooth_data(data, window_size):
    """
    Apply window smoothing to data
    :param data: dict with results
    :param window_size: size of window for smoothing
    :return: smoothed data as dict with key -> (config, smoothed_steps, smoothed_means, smoothed_stds)
    """
    for key, results in data.items():
        config, steps, means, stds = results
        assert (
            len(steps) == len(means) == len(stds)
        ), "Lengths of steps, means, and stds should be the same for smoothing"
        smoothed_steps = []
        smoothed_means = []
        smoothed_stds = []
        for i in range(len(means) - window_size + 1):
            smoothed_steps.append(np.mean(steps[i : i + window_size]))
            smoothed_means.append(np.mean(means[i : i + window_size]))
            smoothed_stds.append(np.mean(stds[i : i + window_size]))
        data[key] = (
            config,
            np.array(smoothed_steps),
            np.array(smoothed_means),
            np.array(smoothed_stds),
        )
    return data


def _get_unique_keys(dicts):
    """
    Get all keys from a list of dicts that do not have identical values across all dicts
    :param dicts: list of dicts
    :return: list of unique keys
    """
    # get all keys across configs
    keys_to_check = set()
    for config in dicts:
        keys_to_check.update(config.keys())

    unique_keys = []
    for key in keys_to_check:
        if key == "hypergroup":
            # skip hypergroup key
            continue
        # add keys that are not in all dicts
        if any(key not in d for d in dicts):
            unique_keys.append(key)
            continue
        # skip keys with dict/ iterable values
        if any(isinstance(d[key], (dict, list)) for d in dicts):
            continue
        # check if value of key is the same for all configs
        if len(set(d[key] for d in dicts)) > 1:
            unique_keys.append(key)
    return unique_keys


def shorten_config_names(data):
    """
    Shorten config names of algorithm to only include hyperparam values that differ across configs
    :param data: dict with results as dict with config_str -> (config, steps, means, stds)
    :return: dict with shortened_config_str -> (config, steps, means, stds)
    """
    configs = [config for config, _, _, _ in data.values()]
    unique_keys_across_configs = _get_unique_keys(configs)

    shortened_data = {}
    for config, steps, means, stds in data.values():
        key_names = []
        for key in unique_keys_across_configs:
            if key not in config:
                continue
            value = config[key]
            if isinstance(value, float):
                value = round(value, 4)
            key_names.append(f"{key}={config[key]}")
        shortened_config_name = "_".join(key_names)
        shortened_data[shortened_config_name] = (config, steps, means, stds)
    return shortened_data


def _sorted_alg_names_by_mean(data):
    """
    Sort alg names by mean value of metric
    :param data: dict with alg names -> (config, steps, means, stds)
    :return: list of sorted alg names
    """
    return sorted(data, key=lambda x: np.mean(data[x][2]), reverse=True)


def _filter_best_per_alg(data):
    """
    Filter data to only contain best performing config per alg
    :param data: dict with key -> (config, steps, means, stds)
    :return: key with highest mean value of means
    """
    means = {key: np.mean(data[key][2]) for key in data}
    return max(means, key=means.get)


def plot_results(data, metric, save_dir, y_min, y_max, log_scale):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    for (_, env, cr, rs), env_data in data.items():
        plt.figure()
        num_plots = 0
        max_label_len = 0
        for alg_name, alg_data in env_data.items():
            if len(alg_data) == 1:
                # plot single curve for algorithm
                key = list(alg_data.keys())[0]
                _, steps, means, stds = alg_data[key]
                plt.plot(steps, means, label=alg_name)
                plt.fill_between(steps, means - stds, means + stds, alpha=ALPHA)
                num_plots += 1
                max_label_len = max(max_label_len, len(alg_name))
            else:
                # plot multiple curves for algorithm, sorted by mean of means
                config_keys_by_performance = _sorted_alg_names_by_mean(alg_data)
                for config_key in config_keys_by_performance:
                    _, steps, means, stds = alg_data[config_key]
                    label = f"{alg_name} ({config_key})"
                    plt.plot(steps, means, label=label)
                    plt.fill_between(steps, means - stds, means + stds, alpha=ALPHA)
                    num_plots += 1
                    max_label_len = max(max_label_len, len(label))
        title = f"{env}"
        title += f" (common rewards; scalarisation {rs})" if cr else " (individual rewards)"
        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel(metric)

        if (
            num_plots > THRESHOLD_FOR_NUM_ALGS_UNTIL_LEGEND_BELOW_PLOT
            or max_label_len > THRESHOLD_FOR_ALG_NAME_LENGTH_UNTIL_LEGEND_BELOW_PLOT
        ):
            # place legend below plot if there are many algos
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        else:
            plt.legend()

        if log_scale:
            plt.yscale("log")
        if y_min is not None or y_max is not None:
            plt.ylim(y_min, y_max)
        if save_dir is not None:
            plt.savefig(save_dir / f"{env}_{metric}_{cr}.pdf", bbox_inches="tight")


def main():
    args = parse_args()
    if args.stage7_json is not None:
        plot_stage7_results(Path(args.stage7_json), Path(args.save_dir))
        return
    data = load_results(args.path, args.metric)
    data = filter_results(data, args.filter_by_algs, args.filter_by_envs)
    data = {
        env_key: {
            alg_name: aggregate_results(alg_data)
            for alg_name, alg_data in env_data.items()
        }
        for env_key, env_data in data.items()
    }
    if args.smoothing_window is not None:
        data = {
            env_key: {
                alg_name: smooth_data(alg_data, args.smoothing_window)
                for alg_name, alg_data in env_data.items()
            }
            for env_key, env_data in data.items()
        }
    data = {
        env_key: {
            alg_name: shorten_config_names(alg_data)
            for alg_name, alg_data in env_data.items()
        }
        for env_key, env_data in data.items()
    }
    if args.best_per_alg:
        best_data = defaultdict(dict)
        for env_key, env_data in data.items():
            for alg_name, alg_data in env_data.items():
                best_config_key = _filter_best_per_alg(alg_data)
                best_data[env_key][alg_name] = {
                    best_config_key: alg_data[best_config_key]
                }
        data = best_data
    plot_results(
        data,
        args.metric,
        Path(args.save_dir),
        args.y_min,
        args.y_max,
        args.log_scale,
    )


if __name__ == "__main__":
    main()
