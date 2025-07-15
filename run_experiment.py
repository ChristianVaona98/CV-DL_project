import os
import random
import subprocess
import time

def main():
    # Generate 5 random seeds
    #seeds = [random.randint(1, 100000) for _ in range(5)]
    seeds = [11598, 49913, 61164, 80541]#,65046
    print(f"Generated seeds list: {seeds}")

    exp_dir = "./experiments"
    #os.makedirs(exp_dir, exist_ok=True)
    
    for seed in seeds:
        # Create seed-specific output directory
        single_exp_output_dir = os.path.join(exp_dir, f"results_seed_{seed}")
        os.makedirs(single_exp_output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running experiment with seed: {seed}")
        print(f"Output directory: {single_exp_output_dir}")
        print(f"{'='*50}")
        
        # Build and execute command
        cmd = f"python main_unet_resnetfpn.py --seed {seed} --output_dir {single_exp_output_dir}"
        print(f"Executing: {cmd}")
        
        start_time = time.time()
        try:
            subprocess.run(cmd, shell=True, check=True)
            status = "SUCCESS"
        except subprocess.CalledProcessError as e:
            status = f"FAILED: {e}"
        
        elapsed = time.time() - start_time
        print(f"\nExperiment completed in {elapsed:.2f} seconds - {status}")

if __name__ == "__main__":
    main()