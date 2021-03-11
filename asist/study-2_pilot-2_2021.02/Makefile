# This Makefile is responsible for pulling messages from the GCS and convert
# them to a format that can be used by the model developed in this project.
# Moreover, there are a specific target for training a model and save its
# parameters to a pre-defined directory under /data. The trained parameters
# will finally be zipped and sent to laplace to be available to an online
# agent.

# The name of the Google Cloud Source folder where the ASIST HSR data resides.
STUDY_ID = study-2_pilot-2_2021.02

# Binaries required for the ASIST evaluation analyses
BUILD_DIR = build

# File that contains configurations of the Saturn map. It is located in the
# directory /data/asist/maps on Laplace, and it will be copied to the folder
# below by this script.
MAP_FILENAME = Saturn_1.0_sm_v1.0.json
MAP_CONFIG_PATH = ../../data/maps/asist/$(MAP_FILENAME)

# Original data from the message bus to be used for training
MESSAGES_DIR = ../../data/asist/$(STUDY_ID)
SPLIT_MESSAGES_DIR = $(MESSAGES_DIR)_split
TRAIN_MESSAGES_DIR = $(SPLIT_MESSAGES_DIR)/train

# File with the trial numbers to use for training
TRAIN_TRIALS_FILE = $(SPLIT_MESSAGES_DIR)/train_list.txt

# Check if trials were defined to training and/or evaluation
SPLIT_DEFINED=$(shell [ -s $(TRAIN_TRIALS_FILE) ] && echo 1 || echo 0 )

# Directory where converted data will reside
TRAIN_SAMPLES_DIR = ../../data/samples/asist/train

# Directory where the trained parameters will be saved
PARAMS_DIR = ../../data/params/asist/$(STUDY_ID)

# Path to the file containing the model definition
MODEL_PATH = ../../models/tomcat-v04.json

# Phony targets
.PHONY: all
.PHONY: build
.PHONY: sync
.PHONY: create_splits
.PHONY: split
.PHONY: convert_train
.PHONY: train

# Builds the system so that the model is retrained if there's a new version of
# the executable available.
build:
	@cd $(BUILD_DIR) && make -j convert
	@cd $(BUILD_DIR) && make -j train

sync:
	@echo ""
	GCS_DIR=$(STUDY_ID) DATA_DIR=$(MESSAGES_DIR) ../../tools/sync_asist_data

