import sys
import json
import re
import numpy as np
from datetime import datetime
from datetime import timedelta
from os import listdir
from os.path import isfile, join


def create_report(eval_filepath, partials_dir, metadata_filepath,
                  horizon, report_filepath):
    """
    This function converts a evaluation file to a report in the format defined
    by DARPA for the asist program.

    :param eval_filepath: filepath of the file with estimates and evaluation
    :param partials_dir: directory where evaluations over model's partials are
    :param metadata_filepath: filepath of the metadata file of the evaluation
    data
    :param horizon: inference horizon for victim rescue
    :param report_filepath: filepath of the final report
    :return:
    """

    evaluations = json.load(open(eval_filepath, "r"))
    metadata = json.load(open(metadata_filepath, "r"))

    num_conditions = 3
    training_condition_estimates = []
    for condition in range(num_conditions):
        training_condition_estimates.append(
            get_estimates(evaluations, 0, "TrainingCondition", condition)
        )

    green_estimates = get_estimates(evaluations, horizon, "Green")
    yellow_estimates = get_estimates(evaluations, horizon, "Yellow")

    with open(report_filepath, "w") as report:
        for i, eval_file in enumerate(metadata["files_converted"]):
            search = re.search("Trial-(\d+)", eval_file["name"], re.IGNORECASE)
            trial = int(search.group(1))

            # Training condition
            report_entry = create_training_condition_entry(
                trial,
                i,
                eval_file["initial_timestamp"],
                training_condition_estimates
            )
            report.write(json.dumps(report_entry) + "\n")

            # Prediction of green victims rescue
            green_entries = create_victim_rescue_entries(
                trial,
                i,
                eval_file["initial_timestamp"],
                green_estimates,
                horizon,
                "Green",
            )
            for report_entry in green_entries:
                report.write(json.dumps(report_entry) + "\n")

            # Prediction of yellow victims rescue
            yellow_entries = create_victim_rescue_entries(
                trial,
                i,
                eval_file["initial_timestamp"],
                yellow_estimates,
                horizon,
                "Yellow",
            )
            for report_entry in yellow_entries:
                report.write(json.dumps(report_entry) + "\n")

    print(
        "Report successfully generated and saved at {}".format(report_filepath)
    )


def get_estimates(evaluations, horizon, node_label, assignment_index=0):
    """
    Extracts estimates for a given estimator, horizon and node label from a set
    of evaluations.

    :param evaluations: json object containing a set of evaluations performed in
     an experiment
    :param horizon: horizon of inference
    :param node_label: label of the node for which evaluations were computed
    :param assignment_index: if estimates where computed for non-binary nodes,
    this indicates the value for which to
    extract the estimates
    :return: list of estimated probabilities
    """

    estimates = [
        estimator["executions"][0]["estimates"][assignment_index]
        for estimator in evaluations["estimation"]["estimators"]
        if estimator["name"] == "sum-product" and estimator[
            "inference_horizon"] == horizon and estimator[
               "node_label"] == node_label][0]

    return np.array(
        [list(map(float, row.split())) for row in estimates.split("\n")]
    )

def get_template_entry(trial):
    """
    Returns a report entry with pre-filled fields that are common to all kinds
    of entries.
    :param trial: trial number
    :return: minimal report entry
    """

    report_entry = {
        "TA": "TA1",
        "Team": "UAZ",
        "AgentID": "ToMCAT",
        "Trial": trial,
    }

    return report_entry


def create_training_condition_entry(
        trial, trial_idx, initial_timestamp, estimates
):
    """
    Creates a report entry for training condition estimates.

    :param trial: trial number
    :param trial_idx: index of the trial in the matrix of evaluation data
    :param initial_timestamp: timestamp when the mission starts
    :param estimates: estimates for training condition
    :return: report entry
    """

    report_entry = get_template_entry(trial)
    last_time_step = estimates[0].shape[1] - 1
    report_entry["Timestamp"] = get_timestamp(
        initial_timestamp, last_time_step
    )
    report_entry[
        "TrainingCondition NoTriageNoSignal"
    ] = estimates[0][trial_idx][-1]
    report_entry[
        "TrainingCondition TriageNoSignal"
    ] = estimates[1][trial_idx][-1]
    report_entry[
        "TrainingCondition TriageSignal"
    ] = estimates[2][trial_idx][-1]
    report_entry["VictimType Green"] = "n.a."
    report_entry["VictimType Yellow"] = "n.a."
    report_entry["VictimType Confidence"] = "n.a."

    return report_entry


def get_timestamp(initial_timestamp, seconds):
    """
    Returns timestamp for a given estimate as the initial timestamp + number of
    seconds elapsed until the estimate.

    :param initial_timestamp: timestamp when the mission starts
    :param seconds: time step when an estimate was computed
    :return: timestamp when an estimate was computed
    """
    timestamp = datetime.strptime(initial_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    timestamp += timedelta(seconds=seconds)
    return datetime.strftime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def create_victim_rescue_entries(
        trial, trial_idx, initial_timestamp, estimates, horizon,
        victim_type
):
    """
    Creates report entries for victim rescue estimates. Each entry represents a
    moment when estimates surpassed the threshold of 0.5 probability,
    indicating that the model is foreseeing a rescue in the next horizon.

    :param trial: trial number
    :param trial_idx: index of the trial in the matrix of evaluation data
    :param initial_timestamp: timestamp when the mission starts
    :param estimates: estimates for victim rescue
    :param horizon: horizon of prediction
    :param victim_type: Green or Yellow
    :return: report entries
    """

    entries = []

    prev_estimate = 0
    for t, estimate in enumerate(estimates[trial_idx]):
        # Only report the estimation prior to start rescuing
        if (estimate >= 0.5) and (prev_estimate < 0.5):
            report_entry = get_template_entry(trial)
            report_entry["Timestamp"] = get_timestamp(initial_timestamp, t)
            report_entry["TrainingCondition NoTriageNoSignal"] = "n.a."
            report_entry["TrainingCondition TriageNoSignal"] = "n.a."
            report_entry["TrainingCondition TriageSignal"] = "n.a."
            report_entry["VictimType Green"] = "n.a."
            report_entry["VictimType Yellow"] = "n.a."
            report_entry["VictimType Confidence"] = "n.a."
            report_entry["Rationale"] = {
                "time_unit": "seconds",
                "time_step_size": 1,
                "horizon_of_prediction": horizon
            }

            if victim_type == "Green":
                report_entry["VictimType Green"] = estimate
            elif victim_type == "Yellow":
                report_entry["VictimType Yellow"] = estimate

            entries.append(report_entry)
        prev_estimate = estimate

    return entries


if __name__ == "__main__":
    eval_filepath = sys.argv[1]
    partials_dir = sys.argv[2]
    metadata_filepath = sys.argv[3]
    horizon = int(sys.argv[4])
    report_filepath = sys.argv[5]
    create_report(eval_filepath, partials_dir, metadata_filepath, horizon,
                  report_filepath)
