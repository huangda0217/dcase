import os
import csv
from typing import Dict, List, Optional
import numpy as np
import scipy
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from inference import ASDPipeline
from dataset.dcase2026task2data import DCSAE2026Task2Data

def save_csv(save_file_path, save_data):

    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def infer_per_machine(
    data_dir: str,
    device: str = "cuda",
    output_dir: str = "results",
    features: Optional[List[str]] = None,
    threshold: float = 0.85,
    mono: bool = False,

):
    os.makedirs(output_dir, exist_ok=True)

    train_data = DCSAE2026Task2Data(data_dir=data_dir, mode="train", mono=mono)
    test_data = DCSAE2026Task2Data(data_dir=data_dir, mode="test", mono=mono)

    machine_types = train_data.machine_type
    print("Machine types:", machine_types)


    train_wavs, train_domains, test_wavs, test_files = [{mt: [] for mt in machine_types} for _ in range(4)]

    for waveform, label, file_name in train_data:
        mt = machine_types[label]
        wav = waveform.numpy().astype(np.float32)
        train_wavs[mt].append(wav)
        train_domains[mt].append("target" if "target" in file_name else "source")

    for waveform, label, file_name in test_data:
        mt = machine_types[label]
        wav = waveform.numpy().astype(np.float32)
        test_wavs[mt].append(wav)
        test_files[mt].append(file_name)

    results = {}
    for mt in machine_types:
        if len(train_wavs[mt]) == 0 or len(test_wavs[mt]) == 0:
            print(f"{mt}: skipped (empty data)")
            continue

        print("\n" + "=" * 60)
        print(f"Processing: {mt}")
        print("=" * 60)

        pipeline = ASDPipeline(device=device, features=features)
        pipeline.fit(train_wavs[mt], np.array(train_domains[mt]))

        train_scores = pipeline.predict(train_wavs[mt])
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(train_scores)
        decision_threshold = scipy.stats.gamma.ppf(threshold, shape_hat, loc=loc_hat, scale=scale_hat)

        scores = pipeline.predict(test_wavs[mt])
        results[mt] = list(zip(test_files[mt], scores.tolist()))

        test_file_names = test_files[mt]
        has_labels = any("normal" in f or "anomaly" in f for f in test_file_names)

        if has_labels:
            fname_score_labels = []
            for fname, score in results[mt]:
                if "normal" in fname:
                    fname_score_labels.append((fname, score, 0))
                elif "anomaly" in fname:
                    fname_score_labels.append((fname, score, 1))

            print(f"{mt}: done, {len(test_wavs[mt])} test files")

            file_names = [x[0] for x in fname_score_labels]
            anomaly_scores_arr = [x[1] for x in fname_score_labels]
            y_true = [x[2] for x in fname_score_labels]
            section_index = os.path.basename(file_names[0]).split("_")[1]

            anomaly_score_csv = os.path.join(output_dir, f"anomaly_score_{mt}_section_{section_index}_test.csv")
            decision_result_csv = os.path.join(output_dir, f"decision_result_{mt}_section_{section_index}_test.csv")

            anomaly_score_list = []
            decision_result_list = []
            for file_name, score, label in fname_score_labels:
                anomaly_score_list.append((os.path.basename(file_name), score))
                decision = 1 if score > decision_threshold else 0
                decision_result_list.append((os.path.basename(file_name), decision))
            save_csv(anomaly_score_csv, anomaly_score_list)
            save_csv(decision_result_csv, decision_result_list)

            y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if "source" in file_names[idx] or y_true[idx] == 1]
            y_pred_s_auc = [anomaly_scores_arr[idx] for idx in range(len(y_true)) if "source" in file_names[idx] or y_true[idx] == 1]
            y_true_t_auc = [y_true[idx] for idx in range(len(y_true)) if "target" in file_names[idx] or y_true[idx] == 1]
            y_pred_t_auc = [anomaly_scores_arr[idx] for idx in range(len(y_true)) if "target" in file_names[idx] or y_true[idx] == 1]

            y_true_s = [y_true[idx] for idx in range(len(y_true)) if "source" in file_names[idx]]
            y_pred_s = [anomaly_scores_arr[idx] for idx in range(len(y_true)) if "source" in file_names[idx]]
            y_true_t = [y_true[idx] for idx in range(len(y_true)) if "target" in file_names[idx]]
            y_pred_t = [anomaly_scores_arr[idx] for idx in range(len(y_true)) if "target" in file_names[idx]]

            auc_s = roc_auc_score(y_true_s_auc, y_pred_s_auc)
            p_auc = roc_auc_score(y_true, anomaly_scores_arr, max_fpr=0.1)
            p_auc_s = roc_auc_score(y_true_s_auc, y_pred_s_auc, max_fpr=0.1)


            tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]).ravel()
            prec_s = tp_s / np.maximum(tp_s + fp_s, 1e-6)
            recall_s = tp_s / np.maximum(tp_s + fn_s, 1e-6)
            f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, 1e-6)

            print(f"AUC (source) : {auc_s:.4f}")
            print(f"pAUC : {p_auc:.4f}")
            print(f"pAUC (source) : {p_auc_s:.4f}")
            print(f"precision (source) : {prec_s:.4f}")
            print(f"recall (source) : {recall_s:.4f}")
            print(f"F1 score (source) : {f1_s:.4f}")

            performance = []

            if len(y_true_t) > 0:
                auc_t = roc_auc_score(y_true_t_auc, y_pred_t_auc)
                p_auc_t = roc_auc_score(y_true_t, y_pred_t, max_fpr=0.1)

                tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]).ravel()
                prec_t = tp_t / np.maximum(tp_t + fp_t, 1e-6)
                recall_t = tp_t / np.maximum(tp_t + fn_t, 1e-6)
                f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, 1e-6)
                print(f"AUC (target) : {auc_t:.4f}")
                print(f"pAUC (target) : {p_auc_t:.4f}")
                print(f"precision (target) : {prec_t:.4f}")
                print(f"recall (target) : {recall_t:.4f}")
                print(f"F1 score (target) : {f1_t:.4f}")
                performance.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

            csv_lines = []
            csv_lines.append(["section", "AUC (Source)", "AUC (Target)", "pAUC (Overall)", "pAUC (Source)", "pAUC (Target)", 
                              "Precision (Source)", "Precision (Target)", "Recall (Source)", "Recall (Target)",
                                "F1 Score (Source)", "F1 Score (Target)"])
            csv_lines.append([f"{mt}_section_{section_index}", auc_s, auc_t, p_auc, p_auc_s, p_auc_t, 
                              prec_s, prec_t, recall_s, recall_t, 
                              f1_s, f1_t])
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), 1e-6), axis=0)
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])
            result_path = os.path.join(output_dir, f"result_{mt}_section_{section_index}_test.csv")
            save_csv(save_file_path=result_path, save_data=csv_lines)

            print(f"AUC (Source): {auc_s:.4f}")
            print(f"AUC (Target): {auc_t:.4f}")
            print(f"pAUC (Source): {p_auc_s:.4f}")
            print(f"pAUC (Target): {p_auc_t:.4f}")
            print(f"pAUC (Overall): {p_auc:.4f}")
        else:
            print(f"{mt}: done, {len(test_wavs[mt])} test files (no labels, anomaly scores only)")
            section_index = os.path.basename(test_file_names[0]).split("_")[1]
            anomaly_score_csv = os.path.join(output_dir, f"anomaly_score_{mt}_section_{section_index}_test.csv")
            anomaly_score_list = [(os.path.basename(f), s) for f, s in results[mt]]
            decision_result_csv = os.path.join(output_dir, f"decision_result_{mt}_section_{section_index}_test.csv")
            decision= [(os.path.basename(f), 1 if s > decision_threshold else 0) for f, s in results[mt]]
            save_csv(anomaly_score_csv, anomaly_score_list)
            save_csv(decision_result_csv, decision)

        print("=" * 50)

    # if output_pkl:
    #     with open(output_pkl, "wb") as f:
    #         pickle.dump(results, f)
    #     print(f"\nResults saved to {output_pkl}")