Scripts & pre trained weights for reproducing our paper results with Flowket.

## Training

```bash
python3 ising_runner.py --gamma 2 train --output_path /path/to/save
python3 ising_runner.py --gamma 2.5 train --output_path /path/to/save
python3 ising_runner.py --gamma 3 train --output_path /path/to/save
python3 ising_runner.py --gamma 3.5 train --output_path /path/to/save
python3 ising_runner.py --gamma 4 train --output_path /path/to/save
```

```bash
python3 heisenberg_runner.py train --output_path /path/to/save
```

## Evaluation

```bash
python3 ising_runner.py --gamma 2 eval --weights_path weights/ising_2.h5
python3 ising_runner.py --gamma 2.5 eval --weights_path weights/ising_2_5.h5
python3 ising_runner.py --gamma 3 eval --weights_path weights/ising_3.h5
python3 ising_runner.py --gamma 3.5 eval --weights_path weights/ising_3_5.h5
python3 ising_runner.py --gamma 4 eval --weights_path weights/ising_4.h5
```

```bash
python3 heisenberg_runner.py eval --weights_path weights/heisenberg.h5
```

### Ising (2^15 sampels)


  | energy | energy per spin | energy variance | \|Mz\|
-- | -- | -- | -- | --
2 | -346.9817926 | -2.409595782 | 0.00125 | 0.783929189
2.5 | -395.6618438 | -2.747651693 | 0.00421 | 0.57235633
3 | -457.0420317 | -3.173902998 | 0.000821 | 0.1622390747
3.5 | -524.5172088 | -3.642480616 | 0.000647 | 0.1106881036
4 | -593.5389339 | -4.121798152 | 0.000777 | 0.09679073758

### Heisenberg (2^19 sampels)

| energy    | energy per spin | energy variance |
|-----------|-----------------|-----------------|
| -251.4536 | -2.514536       | 0.19            |

