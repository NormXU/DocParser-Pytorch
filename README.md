# DocParser: End-to-end OCR-free Information Extraction from Visually Rich Documents

This is an unofficial Pytorch implementation of DocParser.

<figure class="image">
  <img src="doc/encoder_arch.jpeg" alt="{{ encoder architecture }}">
  <figcaption>The architecture of DocParser's Encoder</figcaption>
</figure>

## News
- **Sep 1st**, release the swin-transformer weight [here](https://drive.google.com/file/d/1EmhzE2ZaNOkbS95sHveAt6TMGV9eZWHT/view?usp=drive_link). Please note that this weight is trained with a OCR task and can only be used to initialize the swin-transformer part in the docparser during pretraining. It is NOT intended for fine-tuning in any downstream tasks.
- **July 15th**, update training scripts for Masked Document Reading Task and model architecture.

## How to use
### 1. Set Up Environment
```shell
pip install -r requirements.txt
```

### 2. Prepare Dataset
The dataset should be processed into the following format
```json
{
  "filepath": "path/to/image/folder", // path to image folder
  "filename": "file_name", // file name
  "extract_info": {
    "ocr_info": [
      {
        "chunk": "text1"
      },
      {
        "chunk": "text2"
      },
      {
        "chunk": "text3"
      }
    ]
  } // a list of ocr info of filepath/filename 
}
```
### 3. Start Training
You can start the training from ```train/train_experiment.py``` or

```shell
python train/train_experiment.py --config_file config/base.yaml
```
The training script also support ddp with huggingface/accelerate by
```shell
accelerate train/train_experiment.py --config_file config/base.yaml --use_accelerate True
```
### 4. Notes
The training script currently solely implements the **Masked Document Reading Step** described in the paper. The decoder weights, tokenizer and processor are borrowed from [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base). 

Unfortunately, there is no DocParser pre-training weights publicly available. Simply borrowing weights from Donut-based fails to benefit DocParser on any downstream tasks. But I am working on training a pretraining DocParser based on the two-stage tasks mentioned in the paper recently. Once I successfully complete both the pretraining tasks, and achieve a well-performing model successfully, I intend to make it publicly available on the Huggingface hub. 