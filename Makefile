# Study
STUDY_ID = study-1_2020.08

# Binaries
BUILD_DIR = build

MAP_CONFIG_PATH = data/maps/asist/Falcon_v1.0.json

# Original data from the message bus
MESSAGES_DIR = data/asist/$(STUDY_ID)_split
TRAIN_MESSAGES_DIR = $(MESSAGES_DIR)/train
EVAL_MESSAGES_DIR = $(MESSAGES_DIR)/eval

# Converted data
SAMPLES_DIR = data/samples/asist/$(STUDY_ID)
TRAIN_SAMPLES_DIR = $(SAMPLES_DIR)/train
EVAL_SAMPLES_DIR = $(SAMPLES_DIR)/eval

# Training & evaluation
MODEL_DIR = data/model/asist/$(STUDY_ID)
EVAL_DIR = data/eval/asist/$(STUDY_ID)

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

all: sync split convert train eval report

sync:
	@echo ""
	./tools/sync_asist_data

# This assumes the file eval_list.txt in the $(MESSAGES_DIR) directory contains a list of filenames
# that must be used for evaluation. If there's nothing in this file, all messages will be converted in
# training data.
split:
	@echo ""
	./tools/split_data

# The map configuration file must be downloaded manually from
# https://gitlab.asist.aptima.com/asist/testbed/-/tree/master/Agents/IHMCLocationMonitor/ConfigFolderand
# and be placed in the directory data/maps/asist
convert: sync split
	@echo ""
	@cd $(BUILD_DIR) && make -j TomcatConverter
	@echo "Converting training data..."
	@./$(BUILD_DIR)/bin/TomcatConverter --map_config $(MAP_CONFIG_PATH) --messages_dir $(TRAIN_MESSAGES_DIR) --output_dir $(TRAIN_SAMPLES_DIR)
	@echo "Converting evaluation data..."
	@./$(BUILD_DIR)/bin/TomcatConverter --map_config $(MAP_CONFIG_PATH) --messages_dir $(EVAL_MESSAGES_DIR) --output_dir $(EVAL_SAMPLES_DIR)

train: TomcatASISTFall20 convert
	@echo ""
	@echo "Training model..."
	@./$(BUILD_DIR)/bin/TomcatASISTFall20 --type 0 --data_dir $(TRAIN_SAMPLES_DIR) --model_dir $(MODEL_DIR)

eval: TomcatASISTFall20 train
	@echo ""
	@echo "Evaluating model..."
	@./$(BUILD_DIR)/bin/TomcatASISTFall20 --type 1 --data_dir $(EVAL_SAMPLES_DIR) --model_dir $(MODEL_DIR) --eval_dir $(EVAL_DIR) --horizon $(H)

report: eval
	@echo ""
	@python3 tools/create_asist_report.py $(EVAL_DIR)/evaluations.json $(EVAL_SAMPLES_DIR)/metadata.json $(H) $(EVAL_DIR)/report.txt

TomcatASISTFall20:
	@cd $(BUILD_DIR) && make -j TomcatASISTFall20



