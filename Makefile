# Binaries
BUILD_DIR = build

MAP_CONFIG_PATH = data/maps/asist/Falcon_v1.0.json

# Original data from the message bus
MESSAGES_DIR = data/asist/study-1_2020.08_split
TRAIN_MESSAGES_DIR = $(MESSAGES_DIR)/train
EVAL_MESSAGES_DIR = $(MESSAGES_DIR)/eval

# Converted data
SAMPLES_DIR = data/samples/asist/study-1_2020.08
TRAIN_SAMPLES_DIR = $(SAMPLES_DIR)/train
EVAL_SAMPLES_DIR = $(SAMPLES_DIR)/eval

# Training & evaluation
MODEL_DIR = data/model/asist/study-1_2020.08
EVAL_DIR = data/eval/asist/study-1_2020.08

# Number of seconds ahead predictions are made for victim rescue
H = 1

# Phony targets
.PHONY: all
.PHONY: sync
.PHONY: split
.PHONY: convert
.PHONY: train
.PHONY: eval
.PHONY: report
.PHONY: TomcatASISTFall20

all: sync split convert train eval

sync:
	./tools/sync_asist_data

# This requires the pre creation of the folder $(MESSAGES_DIR) and the file eval_list.txt in it.define
# This file must contain the list of filenames that must be used for evaluation.
split:
	./tools/split_data

# The map configuration file must be downloaded manually from
# https://gitlab.asist.aptima.com/asist/testbed/-/tree/master/Agents/IHMCLocationMonitor/ConfigFolderand
# and be placed in the directory data/maps/asist
convert: sync split
	@cd $(BUILD_DIR) && make -j TomcatConverter
	@echo "Converting training data..."
	@./$(BUILD_DIR)/bin/TomcatConverter --map_config $(MAP_CONFIG_PATH) --messages_dir $(TRAIN_MESSAGES_DIR) --output_dir $(TRAIN_SAMPLES_DIR)
	@echo "Converting evaluation data..."
	@./$(BUILD_DIR)/bin/TomcatConverter --map_config $(MAP_CONFIG_PATH) --messages_dir $(EVAL_MESSAGES_DIR) --output_dir $(EVAL_SAMPLES_DIR)

train: TomcatASISTFall20
	@echo "Training model..."
	@./$(BUILD_DIR)/bin/TomcatASISTFall20 --type 0 --data_dir $(TRAIN_SAMPLES_DIR) --model_dir $(MODEL_DIR)

eval: TomcatASISTFall20
	@echo "Evaluating model..."
	@./$(BUILD_DIR)/bin/TomcatASISTFall20 --type 1 --data_dir $(EVAL_SAMPLES_DIR) --model_dir $(MODEL_DIR) --eval_dir $(EVAL_DIR) --horizon $(H)

report:
	@python3 tools/create_asist_report.py $(EVAL_DIR)/evaluations.json $(EVAL_SAMPLES_DIR)/metadata.json $(H) $(EVAL_DIR)/report.txt

TomcatASISTFall20:
	@cd $(BUILD_DIR) && make -j TomcatASISTFall20



