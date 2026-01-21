import logging
import os
import sys
import time
import gc

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

def release_cache(reason: str | None = None):
    if reason:
        logging.debug(f"Clearing caches: {reason}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    # wandb can return a transient run directory; make sure it exists before checkpointing
    if run is not None and getattr(run, "dir", None):
        os.makedirs(run.dir, exist_ok=True)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]

    record_video_enabled = cfg.get("record_video", False)
    if record_video_enabled:
        cfg.sim.allow_headless_render = True
        cfg.sim.enable_replicator = True

    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int = 0,
        exploration_type: ExplorationType = ExplorationType.MODE,
        tag: str | None = None,
    ):

        base_env.eval()
        env.eval()
        env.set_seed(seed)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        release_cache("evaluate rollout")

        return info

    @torch.no_grad()
    def record_video(
        seed: int = 0,
        exploration_type: ExplorationType = ExplorationType.MODE,
        tag: str | None = None,
    ):
        """Record a rollout video while keeping memory usage low."""
        if not record_video_enabled:
            return {}
        
        base_env.enable_render(True)
        if record_video_enabled:
            base_env.enable_viewport = True
            try:
                base_env._create_viewport_render_product()
            except Exception as exc:
                logging.warning(f"Failed to (re)create viewport for recording: {exc}")
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)
        render_failed = {"error": None}

        def safe_callback(env_cb, *args):
            try:
                return render_callback(env_cb, *args)
            except Exception as exc:
                render_failed["error"] = exc
                return None

        with set_exploration_type(exploration_type):
            try:
                env.rollout(
                    max_steps=base_env.max_episode_length,
                    policy=policy,
                    callback=safe_callback,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
            except Exception as exc:
                render_failed["error"] = render_failed["error"] or exc
            base_env.enable_render(not cfg.headless)
            base_env.enable_viewport = False
            env.reset()
            render_callback.t.close()

        info = {}

        if render_failed["error"] is not None:
            logging.warning(f"Render failed during video recording: {render_failed['error']}")
            render_callback.frames.clear()
            return info

        if len(render_callback.frames) == 0:
            logging.warning(
                "No frames captured during video recording; check render settings (replicator, viewport)."
            )
            return info

        if len(render_callback.frames) > 0 and run is not None and getattr(run, "dir", None):
            video_dir = os.path.join(run.dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            filename = f"eval_{tag if tag is not None else 'final'}.mp4"
            video_path = os.path.join(video_dir, filename)
            try:
                fps = 0.5 / (cfg.sim.dt * cfg.sim.substeps)
                imageio.mimsave(video_path, render_callback.frames, fps=fps)
                info["local_video_path"] = video_path
            except Exception as exc:
                logging.warning(f"Failed to save local eval video: {exc}")

        if len(render_callback.frames) > 0:
            try:
                video_array = render_callback.get_video_array(axes="t c h w")
                if video_array is None:
                    raise ValueError("no valid frames for video")
                info["recording"] = wandb.Video(
                    video_array,
                    fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                    format="mp4",
                )
            finally:
                render_callback.frames.clear()
                if "video_array" in locals():
                    del video_array
        else:
            render_callback.frames.clear()

        release_cache("record_video")

        return info

    num_videos = cfg.get("num_videos", 20)
    video_interval = cfg.get("video_interval", total_frames // num_videos)
    next_video_frame = video_interval if video_interval > 0 else None
    if not record_video_enabled:
        video_interval = None
        next_video_frame = None

    pbar = tqdm(
        collector,
        total=total_frames//frames_per_batch,
        disable=False,           # force display even if stdout is not a TTY
        dynamic_ncols=True,      # adapt width to terminal
        mininterval=0.5,         # update at least every 0.5s
        file=sys.stdout,
    )
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate(tag=str(collector._frames)))
            env.train()
            base_env.train()
            release_cache("post eval")

        if next_video_frame is not None and collector._frames >= next_video_frame:
            logging.info(f"Recording video at {collector._frames} steps.")
            info.update(record_video(tag=str(collector._frames)))
            next_video_frame += video_interval
            release_cache("post video")

        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")
            release_cache("post model")

        run.log(info)
        # use tqdm-aware write to keep progress bar on its own line
        pbar.write(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    release_cache("end of training")
    
    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate(tag="final"))
    release_cache("final eval")
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
