# LevelDP
### About
The code for the paper "LevelDP: A unified expression and implementation of differential privacy and adversarial robustness for structured feature vectors"

### Data set
#### data path
1. mnist: mnist/
2. ustc: ustc/
3. cicids2017: cicids/

#### split dataset
```
cd util/
python MNIST_split.py
python USTC_split.py
python CICIDS_split.py
```
### Environment 
pip install -r requirements.txt
### Run code
cd to the root
1. Train models

Modify training configurations in core/Experiments.py
```
python Train.py
``` 
2. Before you conduct attacks and defenses
```
python ExtractFeature.py
python Delta_f.py
```
3. Membership inference attack
```
python Attack_MIA.py
```
4. Adversarial attack
```
python Attack_Adver.py
```
5. Conduct our defense method for membership inference attack
```
python Defense_MIA.py --dataset "dataset" --model "model" --epsilon "epsilon"
```
6. Conduct our defense method for adversarial attack
```
python Defense_Adver.py --dataset "dataset" --model "model" --epsilon "epsilon"
```
7. Add noise one by one
```
python AddNoiseByFeature_MIA.py --noise_order "noise_order:23/14/random" --epsilon "epsilon"
python AddNoiseByFeature_Adver.py --noise_order "noise_order:12/34/random" --epsilon "epsilon"
```
