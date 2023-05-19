import os
from util.DataSet import process_cicids

os.chdir("../")
process_cicids(name="1.0_test.csv", save_name="test")
process_cicids(name="1.0_train.csv", save_name="train")