from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="engine_project/deployment",     # the local folder containing your files
    repo_id="Praveen-kavedi/engine-maintenance",  # the target HF Space repo
    repo_type="space",                            # dataset, model, or space
    path_in_repo="",                              # optional: subfolder path inside the repo
)
