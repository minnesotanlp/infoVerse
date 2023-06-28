## Data pruning with infoVerse

**Remark**. First, one needs to construct infoVerse following the procedures in `infoverse`. One should move the constructed infoVerse into `outputs` folder. For the tested datasets in our paper, one can download the pre-generated infoverse and selecting indices from the [google drive](https://drive.google.com/file/d/1HwBaEBsbOhFa9fPf2RvCm5xpobTKIW_Z/view?usp=sharing). After that, one can conduct data pruning by controlling `data_ratio` (0.0 to 1.0). Please check out `./run_pruning.sh`. 

**Caution**. The below code should be excuted on the parent folder (`../`)

0. Generating symbolic link to use a customized transformer package.

```
ln -s ../transformers transformers
```
1. Training model with the pruned dataset.

```
python ./data_pruning/train_pruning.py --train_type xxxx_infoverse_dpp --save_ckpt --data_ratio 0.xx --batch_size 16 --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```
