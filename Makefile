CXX ?= clang++

CPPFLAGS := -Iengine/include
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion

BUILD_DIR := build/make
ENGINE_SOURCE := engine/src/position.cpp engine/src/game_history.cpp engine/src/cycle_adjudicator.cpp engine/src/evaluation.cpp engine/src/halfka.cpp engine/src/move_ordering.cpp engine/src/nnue.cpp engine/src/search.cpp engine/src/training_data.cpp engine/src/transposition_table.cpp
CLI_SOURCE := engine/src/main.cpp
DATA_CLI_SOURCE := engine/src/generate_data.cpp
TEST_SOURCE := engine/tests/rules_tests.cpp

.PHONY: all test python-test run clean

all: $(BUILD_DIR)/xiangqi_cli $(BUILD_DIR)/xiangqi_generate_data $(BUILD_DIR)/xiangqi_rules_tests

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/xiangqi_cli: $(ENGINE_SOURCE) $(CLI_SOURCE) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/xiangqi_generate_data: $(ENGINE_SOURCE) $(DATA_CLI_SOURCE) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/xiangqi_rules_tests: $(ENGINE_SOURCE) $(TEST_SOURCE) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

test: $(BUILD_DIR)/xiangqi_rules_tests
	./$(BUILD_DIR)/xiangqi_rules_tests

python-test: $(BUILD_DIR)/xiangqi_cli
	python3 -m unittest discover -s training/tests -v

run: $(BUILD_DIR)/xiangqi_cli
	./$(BUILD_DIR)/xiangqi_cli

clean:
	rm -rf $(BUILD_DIR)
