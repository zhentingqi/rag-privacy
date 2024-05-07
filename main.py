from utils.argparser import get_args
from utils.helpers import fix_seeds, read_json
from modules.LM import LM   
from modules.RALM import RICLM
from modules.Evaluator import Evaluator

import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def main(my_args, llm_args, ric_args, knn_args, training_args, data_args):
    fix_seeds(my_args.my_seed)
    
    if my_args.task == "debug":
        pass
    elif my_args.task == "io":
        assert my_args.api is not None
        assert llm_args.hf_ckpt is not None
        assert llm_args.is_chat_model is not None
        assert data_args.io_input_path is not None
        assert data_args.io_output_root is not None
        
        lm = LM(my_args=my_args, llm_args=llm_args)
        ric_lm = RICLM(ric_args=ric_args, data_args=data_args, lm=lm)
        io_results_dir = os.path.join(data_args.io_output_root, ric_lm.lm.model_name)
        os.makedirs(io_results_dir, exist_ok=True)
        js = read_json(data_args.io_input_path)
        for dict_item in tqdm(js):
            if str(dict_item["id"]) + ".json" in os.listdir(io_results_dir):
                continue
            response = ric_lm.generate(query=dict_item["input"])
            lm_output = response["lm_output"]
            retrieved_docs_str = response["retrieved_docs_str"]
            file_to_save = os.path.join(io_results_dir, str(dict_item["id"]) + ".json")
            with open(file_to_save, "w") as f:
                json.dump({"lm_output": lm_output, "retrieved_docs_str": retrieved_docs_str}, f, indent=4)
    elif my_args.task == "eval":
        assert data_args.eval_input_dir is not None
        assert data_args.eval_output_dir is not None
        os.makedirs(data_args.eval_output_dir, exist_ok=True)
        for model_name in tqdm(os.listdir(data_args.eval_input_dir)):
            if os.path.exists(os.path.join(data_args.eval_output_dir, model_name + ".json")):
                continue
            json_files = [os.path.join(data_args.eval_input_dir, model_name, f) 
                        for f in os.listdir(os.path.join(data_args.eval_input_dir, model_name)) 
                        if f.endswith(".json")]
            predictions_str, references_str = [], []
            for json_file in json_files:
                with open(json_file, "r") as f:
                    js = json.load(f)
                predictions_str.append(js["lm_output"])
                references_str.append(js["retrieved_docs_str"])
            evaluator = Evaluator(predictions_str=predictions_str, references_str=references_str)
            metrics = evaluator.compute_metrics()
            with open(os.path.join(data_args.eval_output_dir, model_name + ".json"), "w") as f:
                json.dump(metrics, f, indent=4)
    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    my_args, llm_args, ric_args, knn_args, training_args, data_args = get_args()
    main(my_args, llm_args, ric_args, knn_args, training_args, data_args)