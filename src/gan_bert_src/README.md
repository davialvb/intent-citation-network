# GAN-BERT

## Train (CLI)

Example:

```bash
python cli_train.py \
  --labeled_csv acl_gan_bert/labeled_scicite.csv \
  --unlabeled_csv acl_gan_bert/unlabeled_scicite.csv \
  --val_csv acl_gan_bert/val_scicite.csv \
  --labels background method result unknown \
  --output_dir weights_run_01
```

## Extra CLIs

### Optuna tuning
Run hyperparameter search (saves best params + a study summary):

```bash
python cli_optuna.py   --labeled_csv acl_gan_bert/labeled_scicite.csv   --unlabeled_csv acl_gan_bert/unlabeled_scicite.csv   --val_csv acl_gan_bert/val_scicite.csv   --labels background method result unknown   --output_dir optuna_runs/run01   --n_trials 30
```

Outputs:
- `optuna_runs/run01/best_params.json`
- `optuna_runs/run01/study_summary.txt`

### t-SNE visualization
Create a 2D/3D t-SNE plot from transformer representations.

Using a training run directory (auto-picks the best transformer checkpoint saved by `cli_train.py`):

```bash
python cli_tsne.py   --csv acl_gan_bert/val_scicite.csv   --text_col text   --label_col label   --model_dir weights_run_01   --model_name allenai/scibert_scivocab_uncased   --n_components 2   --max_points 1000   --output tsne_val.html
```

Or export just the t-SNE coordinates:

```bash
python cli_tsne.py   --csv acl_gan_bert/val_scicite.csv   --text_col text   --label_col label   --model_dir weights_run_01   --model_name allenai/scibert_scivocab_uncased   --output tsne_val.csv
```
