# MedXplain
Enhancing Diagnostic Understanding of X-Rays using Vision Transformer (ViT) from scratch.

## Step To Reproduced
1. Setup venv
2. Setup the dependencies
    ```
    pip install -r requirements.txt
    ```
3. Download the data from `kaggle`
    ```
    import kagglehub

    path = kagglehub.dataset_download("financekim/curated-cxr-report-generation-dataset")

    print("Path to dataset files:", path)
    ```
4. The experiment is conducted in `notebooks/experiment.ipynb`

