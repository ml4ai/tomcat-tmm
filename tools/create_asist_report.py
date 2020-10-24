import sys
import json
import regex as re
import numpy as np


def create_report(eval_filepath, metadata_filepath, horizon, report_filepath):
    """
    This function converts a evaluation file to a report in the format defined
    by DARPA for the asist program.

    :param eval_filepath: filepath of the file with estimates and evaluation
    :param metadata_filepath: filepath of the metadata file of the evaluation data
    :param horizon: inference horizon for victim rescue
    :param report_filepath: filepath of the final report
    :return:
    """

    eval = json.load(open(eval_filepath, 'r'))
    metadata = json.load(open(metadata_filepath, 'r'))

    num_conditions = 3
    training_condition_estimates = []
    for condition in range(num_conditions):
        training_condition_estimates.append(get_estimates(eval, 0, 'TrainingCondition', condition))

    green_estimates = get_estimates(eval, horizon, 'Green')
    yellow_estimates = get_estimates(eval, horizon, 'Yellow')

    with open(report_filepath, 'w') as report:
        for i, eval_file in enumerate(metadata["files_converted"]):
            search = re.search('Trial-(\d+)', eval_file['name'], re.IGNORECASE)
            trial = int(search.group(1))

            # Training condition
            report_entry = create_training_condition_entry(trial, i, eval_file['initial_timestamp'],
                                                           training_condition_estimates)
            report.write(json.dumps(report_entry) + '\n')

            # Prediction of green victims rescue
            green_entries = create_victim_rescue_entries(trial, i, eval_file['initial_timestamp'], green_estimates,
                                                         horizon, 'Green')
            for report_entry in green_entries:
                report.write(json.dumps(report_entry) + '\n')

            # Prediction of yellow victims rescue
            yellow_entries = create_victim_rescue_entries(trial, i, eval_file['initial_timestamp'], yellow_estimates,
                                                         horizon, 'Yellow')
            for report_entry in yellow_entries:
                report.write(json.dumps(report_entry) + '\n')

    print('Report successfully generated and saved at {}'.format(report_filepath))


def get_estimates(evaluations, horizon, node_label, assignment_index=0):
    """
    Extracts estimates for a given estimator, horizon and node label from a set of evaluations.

    :param evaluations: json object containing a set of evaluations performed in an experiment.
    :param horizon: horizon of inference.
    :param node_label: label of the node for which evaluations were computed.
    :param assignment_index: if estimates where computed for non-binary nodes, this indicates the value for which to
    extract the estimates.
    :return: list of estimated probabilities.
    """
    estimates = [estimator['executions'][0]['estimates'][assignment_index] for estimator in
                 evaluations['estimation']['estimators'] if
                 estimator['name'] == 'sum-product' and estimator['inference_horizon'] == horizon and estimator[
                     'node_label'] == node_label][0]

    return np.array([list(map(float, row.split())) for row in estimates.split('\n')])


def get_template_entry(trial):
    report_entry = {"TA": "TA1",
                    "Team": "UAZ",
                    "AgentID": "ToMCAT",
                    "Trial": trial}

    return report_entry


def create_training_condition_entry(trial, trial_idx, initial_timestamp, training_condition_estimates):
    report_entry = get_template_entry(trial)
    report_entry['TrainingCondition NoTriageNoSignal'] = training_condition_estimates[0][trial_idx][-1]
    report_entry['TrainingCondition TriageNoSignal'] = training_condition_estimates[1][trial_idx][-1]
    report_entry['TrainingCondition TriageSignal'] = training_condition_estimates[2][trial_idx][-1]
    report_entry['VictimType Green'] = 'n.a.'
    report_entry['VictimType Yellow'] = 'n.a.'
    report_entry['VictimType Confidence'] = 'n.a.'

    return report_entry


def create_victim_rescue_entries(trial, trial_idx, initial_timestamp, rescue_estimates, horizon, victim_type):
    entries = []

    prev_estimate = 0
    for estimate in rescue_estimates[trial_idx]:
        # Only report the estimation prior to start rescuing
        if (estimate >= 0.5) and (prev_estimate < 0.5):
            report_entry = get_template_entry(trial)
            report_entry['TrainingCondition NoTriageNoSignal'] = 'n.a.'
            report_entry['TrainingCondition TriageNoSignal'] = 'n.a.'
            report_entry['TrainingCondition TriageSignal'] = 'n.a.'
            report_entry['VictimType Green'] = 'n.a.'
            report_entry['VictimType Yellow'] = 'n.a.'
            report_entry['VictimType Confidence'] = 'n.a.'
            report_entry['Rationale'] = {
                'time_unit': "seconds",
                'time_step_size': 1,
                'horizon_of_prediction': horizon
            }

            if victim_type == 'Green':
                report_entry['VictimType Green'] = estimate
            elif victim_type == 'Yellow':
                report_entry['VictimType Yellow'] = estimate

            entries.append(report_entry)
        prev_estimate = estimate

    return entries


if __name__ == "__main__":
    eval_filepath = sys.argv[1]
    metadata_filepath = sys.argv[2]
    horizon = int(sys.argv[3])
    report_filepath = sys.argv[4]
    create_report(eval_filepath, metadata_filepath, horizon, report_filepath)
