DEPO_DIR = ./depo

all: create virtual

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -fr $(DEPO_DIR)
	@echo "\033[92mCleaned up the project.\033[0m"
	@echo "\033[91mDeleted all NPY files in the depo folder.\033[0m"

create:
	@mkdir -p $(DEPO_DIR)
	@echo "\033[92mCreated depo directory.\033[0m"

virtual:
	@if [ ! -d "myenv" ]; then \
		python3.11 -m venv myenv && \
		. myenv/bin/activate && pip install -r ../requirements.txt && \
		echo "\033[92mCreated virtual environment.\033[0m"; \
	else \
		echo "\033[93mVirtual environment already exists.\033[0m"; \
	fi

install:
	pip install -r ../requirements.txt
	@echo "\033[92mInstalled all python packages\033[0m"