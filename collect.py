"""Capture a single-class YOLO dataset from ViZDoom's built-in scenarios."""

import argparse
from collections import Counter
from pathlib import Path
import random

from PIL import Image, ImageDraw
import yaml
import vizdoom as vzd


SCENARIOS = ("basic", "defend_the_center", "defend_the_line")
ENEMIES = {"Cacodemon", "DoomImp", "Demon", "Spectre", "ZombieMan", "ShotgunGuy"}
SPLITS = ("train", "val", "test")


def capture_scenario(args, scenario, scenario_index, output, counts):
    rng = random.Random(args.seed + scenario_index)
    held_out = max(1, args.episodes // 10)
    assignment = ["val", "test"] * held_out
    assignment += ["train"] * (args.episodes - len(assignment))
    rng.shuffle(assignment)
    preview_counts = Counter()
    observed_names = Counter()

    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / f"{scenario}.cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_seed(args.seed + scenario_index)
    game.init()
    try:
        number_of_buttons = game.get_available_buttons_size()
        actions = [[0] * number_of_buttons]
        for index in range(number_of_buttons):
            action = [0] * number_of_buttons
            action[index] = 1
            actions.append(action)

        for episode, split in enumerate(assignment):
            game.new_episode()
            for frame in range(args.frames_per_episode):
                if game.is_episode_finished():
                    break
                state = game.get_state()
                image = Image.fromarray(state.screen_buffer)
                width, height = image.size
                boxes = []
                annotations = []
                for label in state.labels:
                    observed_names[label.object_name] += 1
                    if label.object_name not in ENEMIES:
                        continue
                    x1, y1 = max(0, label.x), max(0, label.y)
                    x2 = min(width, label.x + label.width)
                    y2 = min(height, label.y + label.height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append((x1, y1, x2 - 1, y2 - 1))
                    xywh = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                            (x2 - x1) / width, (y2 - y1) / height)
                    annotations.append("0 " + " ".join(f"{value:.6f}" for value in xywh))

                stem = f"{scenario}_{episode:05d}_{frame:04d}"
                image.save(output / "images" / split / f"{stem}.png")
                if annotations:
                    label_path = output / "labels" / split / f"{stem}.txt"
                    label_path.write_text("\n".join(annotations) + "\n", encoding="utf-8")
                counts[split, "images"] += 1
                counts[split, "boxes"] += len(boxes)
                if boxes and preview_counts[split] < 3:
                    overlay = ImageDraw.Draw(image)
                    for box in boxes:
                        overlay.rectangle(box, outline="lime", width=2)
                    image.save(output / "previews" / f"{split}_{stem}.png")
                    preview_counts[split] += 1

                game.make_action(rng.choice(actions), args.frame_skip)
    finally:
        game.close()
    print(f"{scenario}: observed labels {dict(observed_names)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/doom"))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["basic"])
    parser.add_argument("--episodes", type=int, default=60, help="Episodes per scenario")
    parser.add_argument("--frames-per-episode", type=int, default=60)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 3 or args.frames_per_episode < 1 or args.frame_skip < 1:
        parser.error("Use at least 3 episodes and positive frame counts and frame skip.")
    if not 0 <= args.seed <= 2**32 - len(SCENARIOS):
        parser.error("Seed must fit in ViZDoom's unsigned range after adding scenario IDs.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"Choose an empty output directory to avoid mixing datasets: {output}")
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    (output / "previews").mkdir()

    counts = Counter()
    for scenario in dict.fromkeys(args.scenarios):
        capture_scenario(args, scenario, SCENARIOS.index(scenario), output, counts)
    for split in SPLITS:
        print(f"{split}: {counts[split, 'images']} images, {counts[split, 'boxes']} boxes")
    if any(counts[split, "boxes"] == 0 for split in SPLITS):
        parser.exit(1, "Every split needs enemy labels. Check observed names and collect more episodes.\n")

    dataset = {split: f"images/{split}" for split in SPLITS}
    dataset["names"] = {0: "enemy"}
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    print(f"Dataset: {output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
