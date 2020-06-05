# example run
python main.py --algo_name=BEAR --distance_type=MMD --mode=auto --lagrange_thresh=10.0 --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=20.0 --kernel_type=gaussian --eval_freq=1000 --batch ~/Desktop/LiftDemos/SawyerLiftPhoneSuboptimal/states.hdf5 --name bear_lift_gaussian_20_eval_1000
python main.py --algo_name=BEAR --distance_type=MMD --mode=auto --lagrange_thresh=10.0 --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=20.0 --kernel_type=gaussian --eval_freq=1000 --batch ~/Desktop/RoboTurkPilot/bins-Can/states_top_225.hdf5 --name bear_cans_gaussian_20_eval_1000
python main.py --algo_name=BEAR --distance_type=MMD --mode=auto --lagrange_thresh=10.0 --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=20.0 --kernel_type=gaussian --eval_freq=1000 --batch ~/Desktop/RoboTurkPilot/bins-Can/states_top_225.hdf5 --name bear_cans_all_gaussian_20_eval_1000

python main.py --algo_name=BEAR --distance_type=MMD --mode=auto --lagrange_thresh=10.0 --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=10.0 --kernel_type=laplacian --eval_freq=1000 --batch ~/Desktop/LiftDemos/SawyerLiftPhoneSuboptimal/states.hdf5 --name bear_lift_laplacian_20_eval_1000