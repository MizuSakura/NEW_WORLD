import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_metadata_folder(meta_folder_path: str):
    """
    Scan all YAML files in meta_data folder
    and analyze cumulative_reward and last_error.
    """

    meta_path = Path(meta_folder_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_folder_path} not found")

    yaml_files = sorted(meta_path.glob("episode_*.yaml"))

    if not yaml_files:
        print("No YAML files found.")
        return

    cumulative_rewards = []
    last_errors = []
    episode_ids = []

    for file in yaml_files:
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.unsafe_load(f)

        episode_id = data["episode_info"]["episode_id"]
        cumulative = data["results"]["cumulative_reward"]
        last_error = data["results"]["last_error"]

        episode_ids.append(episode_id)
        cumulative_rewards.append(cumulative)
        last_errors.append(last_error)

    cumulative_rewards = np.array(cumulative_rewards)
    last_errors = np.array(last_errors)



    plt.style.use("seaborn-v0_8-darkgrid")

    plt.figure(figsize=(8,6))

    sc = plt.scatter(
        last_errors,
        cumulative_rewards,
        c=episode_ids,      # ค่าที่ใช้กำหนดสี
        cmap="viridis",     # colormap
        alpha=0.7,
        marker=".",         # รูปแบบ marker
        s=20
    )
    plt.colorbar(sc, label="Episode ID")
#     sns.kdeplot(
#     x=last_errors,
#     y=cumulative_rewards,
#     fill=True,
#     cmap="viridis"
# )

    plt.xlabel("Last Error")
    plt.ylabel("Cumulative Reward")
    plt.title("Last Error vs Cumulative Reward")
    plt.show()
    

    # -------------------------------------------------
    # Summary statistics
    # -------------------------------------------------
    print("\n===== EPISODE ANALYSIS =====")
    print(f"Total episodes: {len(episode_ids)}")

    print("\n--- Cumulative Reward ---")
    print(f"Mean: {np.mean(cumulative_rewards):.3f}")
    print(f"Std : {np.std(cumulative_rewards):.3f}")
    print(f"Max : {np.max(cumulative_rewards):.3f}")
    print(f"Min : {np.min(cumulative_rewards):.3f}")

    print("\n--- Last Error ---")
    print(f"Mean: {np.mean(last_errors):.5f}")
    print(f"Std : {np.std(last_errors):.5f}")
    print(f"Max : {np.max(last_errors):.5f}")
    print(f"Min : {np.min(last_errors):.5f}")
    corr = np.corrcoef(last_errors, cumulative_rewards)[0,1]
    print("Correlation:", corr)

    # -------------------------------------------------
    # Best episodes
    # -------------------------------------------------
    best_reward_idx = np.argmax(cumulative_rewards)
    best_error_idx = np.argmin(last_errors)

    print("\n--- Best Episodes ---")
    print(f"Best Reward Episode: {episode_ids[best_reward_idx]}")
    print(f"Lowest Error Episode: {episode_ids[best_error_idx]}")

    return {
        "episodes": episode_ids,
        "cumulative_rewards": cumulative_rewards,
        "last_errors": last_errors,
    }

if __name__ == "__main__":
    pass
    analyze_metadata_folder(
    r"D:\Project_end\New_world\my_project\logs\image_log\meta_data"
)