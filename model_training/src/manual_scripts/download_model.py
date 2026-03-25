from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OliverHeine/distilbert-base-uncased_train_v2",
    local_dir="../../distilbert_model_android"
)
