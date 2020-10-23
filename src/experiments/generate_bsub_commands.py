def main():
    # experiment for getting diagonal entries of figure 5:
    variable_augmentations = [
        "color_drop",
        "color_jitter",
        "cut_out",
        "flip",
        "random_crop",
        "rotate",
    ]
    base_command = (
        "-W 8:00 -o /cluster/scratch//adahiya/sim_fig5_logs.out -n 16"
        ' -R "rusage[mem=8096, ngpus_excl_p=1]" '
        " -G s_stud_infk python"
        " src/experiments/simclr_experiment.py --resize --crop -epoch 100 -num_workers 16"
    )
    commands = (
        f"bsub -J {variable_augmentations[0]} "
        f"{base_command}"
        f" --{variable_augmentations[0]} "
    )
    for i in range(1, len(variable_augmentations)):
        commands.append(
            f'bsub -J {variable_augmentations[i]} -w "done({variable_augmentations[i-1]})" '
            + base_command
            + f" --{variable_augmentations[i]} "
        )

    f = open("bsub_commands.sh", "w")
    f.write("\n".join(commands))


main()
