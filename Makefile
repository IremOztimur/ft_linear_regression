clean:
	@find ./srcs -type d -name "__pycache__" -exec rm -rf {} +
	@echo "\033[92mCleaned up the project.\033[0m"

virtual:
	@if [ ! -d "myenv" ]; then \
		python3.11 -m venv myenv && \
		. myenv/bin/activate && pip install -r requirements.txt && \
		echo "\033[92mCreated virtual environment.\033[0m"; \
	else \
		echo "\033[93mVirtual environment already exists.\033[0m"; \
	fi

install:
	pip install -r requirements.txt
	@echo "\033[92mInstalled all python packages\033[0m"