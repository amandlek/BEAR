# example run
python main.py --algo_name=BEAR --distance_type=MMD --mode=auto --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=20.0 --kernel_type=gaussian --batch ~/Desktop/LiftDemos/SawyerLiftPhoneSuboptimal/states.hdf5 --name test

python main.py --algo_name=BEAR --distance_type=MMD --mode=auto --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=10.0 --kernel_type=laplacian --batch ~/Desktop/LiftDemos/SawyerLiftPhoneSuboptimal/states.hdf5 --name test