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
    training_condition_intervals = []
    for condition in range(num_conditions):
        training_condition_estimates.append(
            get_estimates(evaluations, 0, "TrainingCondition", condition)
        )
        density_interval = get_density_interval(partials_dir, 0,
                                                "TrainingCondition", condition)
        training_condition_intervals.append(density_interval)

    green_estimates = get_estimates(evaluations, horizon, "Green")
    green_intervals = get_density_interval(partials_dir, horizon, "Green")
    yellow_estimates = get_estimates(evaluations, horizon, "Yellow")
    yellow_intervals = get_density_interval(partials_dir, horizon, "Yellow")

    with open(report_filepath, "w") as report:
        for i, eval_file in enumerate(metadata["files_converted"]):
            search = re.search("Trial-(\d+)", eval_file["name"], re.IGNORECASE)
            trial = int(search.group(1))

            # Training condition
            report_entry = create_training_condition_entry(
                trial,
                i,
                eval_file["initial_timestamp"],
                training_condition_estimates,
                training_condition_intervals,
            )
            report.write(json.dumps(report_entry) + "\n")

            # Prediction of green victims rescue
            green_entries = create_victim_rescue_entries(
                trial,
                i,
                eval_file["initial_timestamp"],
                green_estimates,
                green_intervals,
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
                yellow_intervals,
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


def get_density_interval(partials_dir, horizon, node_label, assignment_index=0):
    """
    Extracts minimum and maximum estimates for a given estimator, horizon and node label from a set
    of evaluations.

    :param partials_dir: directory where partial evaluation files are
    :param horizon: horizon of inference
    :param node_label: label of the node for which evaluations were computed
    :param assignment_index: if estimates where computed for non-binary nodes,
    this indicates the value for which to
    extract the estimates
    :return: tuple (min estimates, max estimates)
    """
    all_estimates = []
    for eval_filename in listdir(partials_dir):
        eval_filepath = join(partials_dir, eval_filename)
        if isfile(eval_filepath):
            evaluations = json.load(open(eval_filepath, "r"))
            all_estimates.append(
                get_estimates(evaluations, horizon, node_label,
                              assignment_index))

    min_estimates = np.min(all_estimates, axis=0)
    max_estimates = np.max(all_estimates, axis=0)

    return min_estimates, max_estimates


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
        trial, trial_idx, initial_timestamp, estimates, intervals
):
    """
    Creates a report entry for training condition estimates.

    :param trial: trial number
    :param trial_idx: index of the trial in the matrix of evaluation data
    :param initial_timestamp: timestamp when the mission starts
    :param estimates: estimates for training condition
    :param intervals: min and max estimates for training condition
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

    min_estimate_0 = intervals[0][0][trial_idx][-1]
    max_estimate_0 = intervals[0][1][trial_idx][-1]
    min_estimate_1 = intervals[1][0][trial_idx][-1]
    max_estimate_1 = intervals[1][1][trial_idx][-1]
    min_estimate_2 = intervals[2][0][trial_idx][-1]
    max_estimate_2 = intervals[2][1][trial_idx][-1]
    report_entry["Rationale"] = {
        "DensityInterval NoTriageNoSignal": "[{}, {}]".format(min_estimate_0,
                                                              max_estimate_0),
        "DensityInterval TriageNoSignal": "[{}, {}]".format(min_estimate_1,
                                                            max_estimate_1),
        "DensityInterval TriageSignal": "[{}, {}]".format(min_estimate_2,
                                                          max_estimate_2),
    }

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
        trial, trial_idx, initial_timestamp, estimates, intervals, horizon,
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
    :param intervals: min and max estimates for training condition
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
            min_estimate = intervals[0][trial_idx][t]
            max_estimate = intervals[1][trial_idx][t]
            report_entry["Rationale"] = {
                "time_unit": "seconds",
                "time_step_size": 1,
                "horizon_of_prediction": horizon,
                "density_interval": "[{}, {}]".format(min_estimate,
                                                      max_estimate)
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
