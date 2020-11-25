This file contains the commands used to do each experiment.
# baseline experiment
python src/experiments/baseline_experiment.py -crop -resize -rotate

# Experiment 1
Determining how long should the simclr model with contrastive loss be trained to obtain well trained encoder for semi supervised experiment.
## a
```python src/experiments/simclr_experiment.py --resize --crop --gaussian_blur --rotate --color_jitter --color_drop --cut_out  -epochs 1000 -batch_size 256```
## b
```python src/experiments/simclr_experiment.py --resize --crop --random_crop --gaussian_blur --color_jitter --color_drop --cut_out  -epochs 1000 -batch_size 128```
## c (minibatch 256)
```python src/experiments/simclr_experiment.py --resize --crop --random_crop --gaussian_blur --color_jitter --color_drop --cut_out  -epochs 1000 -batch_size 1024```
## d (minibatch 512)
```python src/experiments/simclr_experiment.py --resize --crop --random_crop --gaussian_blur --color_jitter --color_drop --cut_out  -epochs 1000 -batch_size 2048```
## e
In this experiement baseline model was trained with fixed resent  weights. It is more like sanity check.
```python src/experiments/experiment1_e.py```

#### Following commands are run in leonhard
bsub -J 24 -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint  24 
bsub -J 49 -w "done('24')" -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint 49 
bsub -J 74 -w "done('49')" -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint 74 
bsub -J 99 -w "done('74')" -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint 99 
bsub -J 124 -w "done('99')" -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint 124 
bsub -J 149 -w "done('124')" -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint 149 
bsub -J 174 -w "done('149')" -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint 174 
bsub -J 199 -w "done('174')" -W 4:00 -o /cluster/scratch//adahiya/experiment1a_logs.out -n 16 -R "rusage[mem=4096, ngpus_excl_p=1]"  -G s_stud_infk python src/experiments/experiment1_a.py -checkpoint 199  

#### Following commands are run in Lab pc
python src/experiments/experiment1_a.py -checkpoint 224


### Experiment 4

The pretraining phase. The images are cropped automatically with same jitter wherever necessary. They are not rotated if not specified.
1. Rotation: Both the images are sampled and rotated. The crop jitter is kept same.
```python src/experiments/experiment4_pretraining.py --rotate --resize  -batch_size 2048  -epochs 100```
2. Blur: Gaussian blur is applied randomly to both the images. Jitter is same
```python src/experiments/experiment4_pretraining.py --gaussian_blur --resize  -batch_size 2048  -epochs 100```
3. Random crop: Cropping is done randomly in the crop box containing the hand. Jitter is random
```python src/experiments/experiment4_pretraining.py --crop --random_crop  --resize  -batch_size 2048  -epochs 100```
4. Standard crop: The hand is cropped with random jitter.
```python src/experiments/experiment4_pretraining.py --crop  --resize  -batch_size 2048  -epochs 100```
5. Color jitter:  The hsv values are jittered with crop jitter same.
```python src/experiments/experiment4_pretraining.py --color_jitter  --resize  -batch_size 2048  -epochs 100```
6. Color drop: Image is converted to gray scale randomly. crop Jitter is kept same.
```python src/experiments/experiment4_pretraining.py --color_drop  --resize  -batch_size 2048  -epochs 100```




python src/experiments/experiment4_pretraining.py --rotate --resize  -batch_size 2048  -epochs 100
python src/experiments/experiment4_pretraining.py --rotate --resize  -batch_size 2048  -epochs 100
