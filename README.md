# Robust Explanations of Graph Neural Networks via Graph Curvatures
This is the Pytorch Implementation of [Robust Explanations of Graph Neural Networks via Graph Curvatures](https://openreview.net/pdf?id=48L3BEtH8w)

# Train a Model

Supports GNN model training for three task types:
- **Node Classification**: Cora, CiteSeer, PubMed
- **Link Prediction**: UCI,BC-OTC,BC-ALPHA
- **Graph Classification**: TUDataset (PROTEINS, IMDB-BINARY, etc.)
  
You need to train a model first. To train the model, use th following command.
```bash
# Node Classification
python train_GCN.py --task node_classification --dataset Cora --normalize_adj

# Link Prediction
python train_GCN.py --task link_prediction --dataset UCI

# Graph Classification
python train_GCN.py --task graph_classification --dataset IMDB-BINARY
```

# Provide an explanation with graph curvature

Provides explanation for GNN model supporting multiple explanation methods and enhancing robustness of explanations through Ollivier-Ricci curvature and effective resistance.

We consider the following explanation methods: **DeepLIFT**, **FlowX**, **GNN-LRP**, **GNNExplainer**, **PGExplainer**, and **Convex**

#### 1. TRAIN Mode - Filter training samples (nodes/edges/graphs) that are suitable for lambda optimization.
```bash
# Node Classification
python node_explain.py --dataset Cora --method deeplift --run_mode train

# Link Prediction
python link_explain.py --dataset UCI --method deeplift --run_mode train

# Graph Classification
python graph_explain.py --dataset IMDB-BINARY --method deeplift --run_mode train
```

#### 2. SELECT Mode - Use Optuna to find optimal lambda values that improve explanation robustness when combined with graph curvature.
```bash
# Node Classification
python node_explain.py --dataset Cora --method deeplift --run_mode select \
--curvature_type ricci --lambda_min 0.0 --lambda_max 0.1 --n_trials 50

# Link Prediction
python link_explain.py --dataset UCI --method deeplift --run_mode select \
    --curvature_type ricci --lambda_min 0.0 --lambda_max 0.1 --n_trials 50

# Graph Classification
python graph_explain.py --dataset IMDB-BINARY --method deeplift --run_mode select \
    --curvature_type resistance --lambda_min 0.0 --lambda_max 0.1 --n_trials 50
```

#### 3. VAL Mode - Evaluate the curvature-enhanced explanations on validation samples using the optimized lambda values.
```bash
# Node Classification
python node_explain.py --dataset Cora --method deeplift --run_mode val

# Link Prediction
python link_explain.py --dataset UCI --method deeplift --run_mode val

# Graph Classification
python graph_explain.py --dataset IMDB-BINARY --method deeplift --run_mode val
```

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name | Cora/Citeseer/Pheme/UCI/BC-OTC/BC-Alpha/MUTAG/IMDB-BINARY/PROTEINS |
| `--method` | Explanation method | DeepLIFT/FlowX/GNN-LRP/GNNExplainer/PGExplainer/Convex |
| `--run_mode` | Run mode (train/select/val) | select |
| `--curvature_type` | Curvature type (ricci/resistance) | ricci |
| `--sparsity` | Fraction of edges to select | 0.1 |
| `--lambda_min` | Minimum lambda for optimization | 0.0 |
| `--lambda_max` | Maximum lambda for optimization | 0.1 |
| `--n_trials` | Number of Optuna trials | 50 |
| `--num_remove_val_ratio` | Ratio of edges to remove in perturbation | 0.1 |
| `--num_add_val_ratio` | Ratio of edges to add in perturbation | 0.1 |
| `--normalize_adj` | Whether to normalize adjacency matrix | True/False |
| `--seed` | Random seed | 42 |

### Citation
If you use this code in your research, please cite:
```bash
@article{liurobust,
  title={Robust Explanations of Graph Neural Networks via Graph Curvatures},
  author={Liu, Yazheng and Zhang, Xi and Xie, Sihong and Xiong, Hui},
  conference={The Thirty-ninth Annual Conference on Neural Information Processing Systems}。
  year={2025},
}
```
