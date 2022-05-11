## Acknowledgment

The code is written based on the [Zhang et al's code](https://github.com/chuanting/FedDA), and still under polishment.

## Download the dataset

The dataset can be manually downloaded from [Google Drive](https://drive.google.com/drive/folders/1_F1-x3Gff2tO3u3MxLDyfYYbGhCM0kUZ?usp=sharing)

After downloading the .h5 files, put those files into a directory named dataset and put the directory under the main directory before running the code.

## Train centralized model
```commandline
python centralized.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --epochs 10 --batch_size 50
```
Centrailized File: milano.h5 Type: sms MSE: 0.3171 MAE: 0.3223, NRMSE: 0.0908

#### Uniform noise for comparison
```commandline
python centralized.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --attack_epsilon 0.2 --attack_optimizer uniform
```

#### Data poisoning against centralized model training
```commandline
python centralized.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --attack_epsilon 0.2 --num_ensemble 2 --attack_rounds 30 --epochs 10 --batch_size 50 --attack_lr 10.0 --mask_prob 0.8 --surrogate_model lstm
```

#### Apply defense to data poisoning (Data Sanitization, Randomized Smoothing)

```commandline
python centralized.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --attack_epsilon 0.2 --num_ensemble 2 --attack_rounds 30 --epochs 10 --batch_size 50 --attack_lr 10.0 --mask_prob 0.8 --surrogate_model lstm --apply_defense sphere_sani
```

```commandline
python centralized.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --attack_epsilon 1.0 --num_ensemble 2 --attack_rounds 20 --epochs 10 --batch_size 50 --attack_lr 10.0 --mask_prob 0.8 --surrogate_model lstm --apply_defense adj_sani
```

```commandline
python centralized.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --attack_epsilon 1.0 --num_ensemble 2 --attack_rounds 20 --epochs 10 --batch_size 50 --attack_lr 10.0 --mask_prob 0.8 --surrogate_model lstm --apply_defense rand
```



## Train FedAvg model
```commandline
python fed_avg.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1
```
Type: sms MSE: 0.3744 MAE: 0.3386, NRMSE: 0.0955

#### Model poisoning against FedAvg
```commandline
python fed_avg.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison
```

#### Apply defenses (Multi Krum, Trimmed Mean)
```commandline
python fed_avg.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --apply_defense multi_krum
```

```commandline
python fed_avg.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --apply_defense trimmed_mean
```

```commandline
python fed_avg.py --file milano.h5 --type sms --lr 1e-2 --frac 0.1 --bs 100 --opt 'sgd' --out_dim 1 --poison --apply_defense median
```



