from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="output/huawei-noah_TinyBERT_General_4L_312D",
    repo_id="OliverHeine/huawei-noah_TinyBERT_General_4L_312D_train_v2",
    repo_type="model",
    token="<REPLACE_TOKEN>"
)