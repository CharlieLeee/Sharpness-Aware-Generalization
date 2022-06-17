python run.py --batch_size 128 --seed 1 --epochs 200 --baseoptim 'sgd' --secoptim 'asam' --model 'resnet18' --norm_type 'none' --no-batchnorm --lr 0.01 --wd 0.0005 --rho 0.5 --momentum 0.9 --cosaneal
python run.py --batch_size 128 --seed 1 --epochs 200 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'none' --no-batchnorm --lr 0.01 --wd 0.0005 --rho 0.05 --momentum 0.9 --cosaneal

python run.py --batch_size 128 --seed 2 --epochs 200 --baseoptim 'sgd' --secoptim 'asam' --model 'resnet18' --norm_type 'none' --no-batchnorm --lr 0.01 --wd 0.0005 --rho 0.5 --momentum 0.9 --cosaneal
python run.py --batch_size 128 --seed 2 --epochs 200 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'none' --no-batchnorm --lr 0.01 --wd 0.0005 --rho 0.05 --momentum 0.9 --cosaneal

python run.py --batch_size 128 --seed 3 --epochs 200 --baseoptim 'sgd' --secoptim 'asam' --model 'resnet18' --norm_type 'none' --no-batchnorm --lr 0.01 --wd 0.0005 --rho 0.5 --momentum 0.9 --cosaneal
python run.py --batch_size 128 --seed 3 --epochs 200 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'none' --no-batchnorm --lr 0.01 --wd 0.0005 --rho 0.05 --momentum 0.9 --cosaneal