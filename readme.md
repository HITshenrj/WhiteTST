### Usage

1. Install Python 3.8. For convenience, execute the following command.
   pip install -r requirements.txt

3. Prepare Data. You can obtain the well pre-processed datasets from [Google Drive] or [Baidu Drive], Then place the downloaded data in the folder./dataset.

4. Train and evaluate model. You can reproduce the experiment results as the following examples:
   python run_longExp.py --is_training 1 --model WhiteTST --data ETTh1 --root_path ./dataset/ETT-small --data_path ETTh1.csv

5. Develop your own model.
Add the model file to the folder ./models. You can follow the ./models/Transformer.py.
Include the newly added model in the Exp_Basic.model_dict of ./exp/exp_basic.py.
Create the corresponding scripts under the folder ./scripts.

### Acknowledgement
Thanks https://github.com/thuml/Time-Series-Library
