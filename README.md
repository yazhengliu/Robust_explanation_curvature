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

#### 1. TRAIN Mode - Filter Training Nodes
```bash
python explain.py --run_mode train --dataset Cora --method gnnlrp
```
The training nodes save to `train_node.json`

#### 2. VAL Mode - Filter Validation Nodes
```bash
python explain.py --run_mode val --dataset Cora --method gnnlrp
```
The validation nodes save to `val_node.json`

#### 3. SELECT Mode - Optimize Lambda Parameter
```bash
python explain.py --run_mode select --dataset Cora --method gnnlrp --n_trials 50
```
