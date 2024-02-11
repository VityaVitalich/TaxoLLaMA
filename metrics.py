import os
import yaml

with open(r"./configs/metrics.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

import sys
import pandas as pd
import numpy as np
import pickle


from pipeline_src.metrics.metrics import Metric
from pipeline_src.dataset.dataset import HypernymDataset
from transformers import AutoTokenizer

from pipeline_src.dataset.prompt_schemas import (
    hypo_term_hyper,
    predict_child_from_2_parents,
    predict_child_from_parent,
    predict_child_with_parent_and_grandparent,
    predict_children_with_parent_and_brothers,
    predict_parent_from_child_granparent,
    predict_parent_from_child,
    predict_multiple_parents_from_child,
)


def get_intersect(gold, pred):
    return list(set(pred).intersection(set(gold)))


if __name__ == "__main__":
    test_path = params_list["TEST_DATA_PATH"][0]
    saving_path = params_list["OUTPUT_NAME"][0]
    save_examples = params_list["SAVE_EXAMPLES"][0]
    save_examples_path = params_list["SAVE_EXAMPLES_PATH"][0]
    ids_to_use = params_list["IDS_TO_USE"][0]
    decoding = params_list["DECODING"][0]

    df = pd.read_pickle(test_path)

    transforms = {
        "only_child_leaf": predict_child_with_parent_and_grandparent, 
        "only_leafs_all": predict_child_from_parent,
        "only_leafs_divided": predict_children_with_parent_and_brothers,
        "leafs_and_no_leafs": predict_child_from_parent,
        "simple_triplet_grandparent": predict_parent_from_child_granparent,
        "simple_triplet_2parent": predict_child_from_2_parents,
        "predict_hypernym": predict_parent_from_child,
        "predict_multiple_hypernyms": predict_multiple_parents_from_child,
    }

    with open(saving_path, "rb") as fp:
        all_preds = pickle.load(fp)

    # with open("./babel_datasets/v2_wnet_test_hard_ids.pickle", "rb") as fp:
    #     hard_ids = pickle.load(fp)

    if ids_to_use == "easy":
        all_preds = [elem for i, elem in enumerate(all_preds) if i not in hard_ids]
        df = [elem for i, elem in enumerate(df) if i not in hard_ids]
        write_log_path = save_examples_path + "/easy_metrics.txt"

    elif ids_to_use == "hard":
        all_preds = [elem for i, elem in enumerate(all_preds) if i in hard_ids]
        df = [elem for i, elem in enumerate(df) if i in hard_ids]
        write_log_path = save_examples_path + "/hard_metrics.txt"
    else:
        write_log_path = save_examples_path + "/metrics.txt"

    if isinstance(all_preds[0][0], list):
        flat_list = [item for sublist in all_preds for item in sublist]
        all_preds = flat_list

    all_labels = []
    all_terms = []
    cased = {"all_hyponyms": {"pred": [], "label": [], "term": []}}

    for i, elem in enumerate(df):
        try:
            all_preds[i]
        except IndexError:
            continue

        case = elem["case"]
        processed_term, target = transforms[case](elem)
        all_labels.append(target)
        all_terms.append(processed_term)

        if not case in cased.keys():
            cased[case] = {"pred": [], "label": [], "term": []}

        cased[case]["pred"].append(all_preds[i])
        cased[case]["label"].append(target)
        cased[case]["term"].append(processed_term)

        if case in ("leafs_and_no_leafs", "only_leafs_all", "only_child_leaf"):
            cur_case = "all_hyponyms"
            cased[cur_case]["pred"].append(all_preds[i])
            cased[cur_case]["label"].append(target)
            cased[cur_case]["term"].append(processed_term)

    print("total preds:" + str(len(all_preds)))
    print("total labels:" + str(len(all_labels)))
    metric_counter = Metric(all_labels, all_preds, decoding=decoding)
    mean_cased = metric_counter.get_metrics()

    cased_metrics = {}
    for key in cased.keys():
        metric_counter = Metric(cased[key]["label"], cased[key]["pred"])
        res = metric_counter.get_metrics()
        cased_metrics[key] = res

    if not os.path.exists(save_examples_path):
        print("path {} do not exist".format(save_examples_path))
        os.mkdir(save_examples_path)

    with open(write_log_path, "w") as f:
        df = pd.concat(
            [pd.DataFrame(cased_metrics), (pd.DataFrame(mean_cased, index=["mean"]).T)],
            axis=1,
        )
        f.write(df.to_string())
        print("metrics written in file")

    if save_examples:
        if not os.path.exists(save_examples_path):
            print("path {} do not exist".format(save_examples_path))
            os.mkdir(save_examples_path)

        metric_counter = Metric(all_labels, all_preds)

        for key in cased.keys():
            n = len(cased[key]["pred"])

            total_str = ""
            for i in range(n):
                preds = cased[key]["pred"][i]
                m = len(preds)
                for j in range(m):
                    pred = preds[j]
                    gold = cased[key]["label"][i]
                    term = cased[key]["term"][i]

                    res = metric_counter.get_one_prediction(
                        gold, pred, metric_counter.default_metrics(), limit=50
                    )
                    intersect = get_intersect(
                        metric_counter.get_hypernyms(gold),
                        metric_counter.get_hypernyms(pred),
                    )

                    res_str = ""
                    for metric_key in res.keys():
                        res_str += " " + str(metric_key) + " " + str(res[metric_key])

                    total_str += (
                        term
                        + "\n\n"
                        + "predicted: "
                        + pred
                        + "\n \n"
                        + "true: "
                        + gold
                        + "\n \n"
                        + "intersection: "
                        + ",".join(intersect)
                        + "\n\n"
                        + "metrics: "
                        + res_str
                        + " \n\n"
                        + "=" * 10
                        + "\n"
                    )

            file_name = save_examples_path + "/" + str(key) + ".txt"
            with open(file_name, "w") as f:
                f.write(total_str)
