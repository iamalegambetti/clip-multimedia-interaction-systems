# Clip Multimedia Interaction Systems
This work serves as the evaluation project for the subject Multimedia Interaction and Systems for the Academic year 2023/2024. 

## Guidelines

### Optimization
To train and evaluate CLIP models:

- Fully-train CLIP from scratch: 
`experiments/CLIP/paper_replication/train.py` 

- Fine-tune CLIP: 
`experiments/CLIP/paper_replication/fine_tune.py` 

- Zero-shot CLIP evaluation:
`experiments/CLIP/paper_replication/zero_shot_benchmark.py`

### Dataset & Embeddings
Dataset & Saved Embeddings are downloadable [here](https://drive.google.com/drive/folders/1kHT6J1kCezuNgtcCDDi8sjwEHIbR8w_p?usp=drive_link).
Embeddings should be stored in their respective folder.
Images are available for download [here](https://drive.google.com/file/d/1TtZGLRZ3sa1qwqKJ5NviZz7_PUFbr4gf/view?usp=drive_link).

### Case Study 
The case study can be replicated by configuring and running the file:

`experiments/run_experiments.py`