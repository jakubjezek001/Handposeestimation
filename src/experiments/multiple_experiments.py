import os
import subprocess


def main():
    # experiment for getting diagonal entries of figure 5:
    variable_augmentations = [
        "color_drop",
        "color_jitter",
        "cut_out",
        "flip",
        "random_crop",
        "rotate" "gaussian_blur",
    ]
    for aug in variable_augmentations:
        experiment = subprocess.Popen(
            [
                (
                    "python "
                    "src/experiments/simclr_experiment.py "
                    "--resize "
                    "--crop "
                    "--gaussian_blur "
                    f"--{aug} "
                    "-epochs "
                    "100"
                )
            ],
            shell=True,
        )
        experiment.wait()
        # stdout_value, stderr_value = experiment.communicate()
        # print(stderr_value, stdout_value)


main()
