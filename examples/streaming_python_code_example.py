#!/usr/bin/env python3
"""
Example script demonstrating how to stream a large Python code block (2 pages worth)
using the rich library. This shows a comprehensive Python class with various
programming concepts and patterns.
"""

import time
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.syntax import Syntax


def stream_python_code_with_live(console, code_text, chunk_size=15, delay=0.05, window_lines=16):
    """
    Stream Python code with a sliding window of the last N lines, then show full result.

    Args:
        console: Rich Console instance
        code_text: The Python code text to stream
        chunk_size: Number of characters per chunk (default: 15)
        delay: Delay between chunks in seconds (default: 0.05)
        window_lines: Number of lines to show in the sliding window (default: 16)
    """
    accumulated_text = ""
    lines = []

    # Extract the actual Python code (remove the markdown code block markers)
    code_content = code_text.replace("```python\n", "").replace("```", "")

    # Create a console with highlighting disabled for streaming
    plain_console = Console(highlight=False)

    with Live(console=plain_console, refresh_per_second=20,
              vertical_overflow="ellipsis") as live:
        for i in range(0, len(code_content), chunk_size):
            chunk = code_content[i:i + chunk_size]
            accumulated_text += chunk

            # Split into lines and keep only the last window_lines
            current_lines = accumulated_text.split('\n')
            if len(current_lines) > window_lines:
                display_lines = current_lines[-window_lines:]
                # Add ellipsis to indicate there's more content above
                display_text = "...\n" + '\n'.join(display_lines)
            else:
                display_text = '\n'.join(current_lines)

            # Update with plain text (no syntax highlighting during streaming)
            live.update(display_text)
            time.sleep(delay)

        # After streaming is complete, show the full syntax-highlighted version
        time.sleep(0.5)  # Brief pause before final display
        syntax = Syntax(code_content, "python", theme="default", background_color=None)
        live.update(syntax)


# Python code block for streaming demonstration
python_code = """```python
#!/usr/bin/env python3
\"\"\"
Data Processing Example

A simple but comprehensive data processing class demonstrating
various Python programming concepts and patterns.
\"\"\"

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class DataFormat(Enum):
    \"\"\"Supported data formats.\"\"\"
    JSON = "json"
    CSV = "csv"
    XML = "xml"


@dataclass
class ProcessingMetrics:
    \"\"\"Track processing performance.\"\"\"
    records_processed: int = 0
    errors_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        \"\"\"Calculate processing duration.\"\"\"
        return self.end_time - self.start_time


class DataValidator(ABC):
    \"\"\"Abstract base class for data validators.\"\"\"

    @abstractmethod
    def validate(self, data: Any) -> bool:
        \"\"\"Validate the given data.\"\"\"
        pass


class SchemaValidator(DataValidator):
    \"\"\"Validator that checks data against a schema.\"\"\"

    def __init__(self, required_fields: List[str]):
        self.required_fields = required_fields

    def validate(self, data: Any) -> bool:
        if isinstance(data, dict):
            return all(field in data for field in self.required_fields)
        return False


class DataProcessor:
    \"\"\"Main data processing class.\"\"\"

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.metrics = ProcessingMetrics()
        self.validators: List[DataValidator] = []

    def add_validator(self, validator: DataValidator) -> None:
        \"\"\"Add a data validator.\"\"\"
        self.validators.append(validator)

    def validate_data(self, data: Any) -> None:
        \"\"\"Validate data using all validators.\"\"\"
        for validator in self.validators:
            if not validator.validate(data):
                raise ValueError(f"Validation failed for data: {data}")

    async def process_data_async(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Process data asynchronously.\"\"\"
        import time
        self.metrics.start_time = time.time()
        processed_data = []

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            processed_batch = await self._process_batch_async(batch)
            processed_data.extend(processed_batch)

            # Log progress
            self.logger.info(f"Processed batch {i//self.batch_size + 1}")

        self.metrics.end_time = time.time()
        self.metrics.records_processed = len(processed_data)
        return processed_data

    async def _process_batch_async(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Process a single batch.\"\"\"
        processed_batch = []

        for record in batch:
            try:
                self.validate_data(record)
                transformed = self._transform_record(record)
                processed_batch.append(transformed)
            except Exception as e:
                self.logger.error(f"Error processing record: {e}")
                self.metrics.errors_count += 1

        return processed_batch

    def _transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Transform a single record.\"\"\"
        transformed = record.copy()

        # Example transformations
        for key, value in transformed.items():
            if isinstance(value, str):
                transformed[key] = value.strip().lower()
            elif isinstance(value, (int, float)):
                transformed[key] = round(value, 2)

        return transformed

    def get_metrics_summary(self) -> Dict[str, Any]:
        \"\"\"Get processing metrics summary.\"\"\"
        return {
            'duration': self.metrics.duration,
            'records_processed': self.metrics.records_processed,
            'errors_count': self.metrics.errors_count,
            'success_rate': (self.metrics.records_processed - self.metrics.errors_count)
                          / max(self.metrics.records_processed, 1) * 100
        }


# Example usage
async def main():
    \"\"\"Example usage of the DataProcessor.\"\"\"
    # Create processor
    processor = DataProcessor(batch_size=50)

    # Add validator
    validator = SchemaValidator(['id', 'name', 'value'])
    processor.add_validator(validator)

    # Sample data
    sample_data = [
        {'id': 1, 'name': 'Item 1', 'value': 10.5},
        {'id': 2, 'name': 'Item 2', 'value': 20.3},
        {'id': 3, 'name': 'Item 3', 'value': 15.7}
    ]

    # Process data
    try:
        result = await processor.process_data_async(sample_data)
        print("Processing completed!")
        print(f"Metrics: {processor.get_metrics_summary()}")
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
```"""


def main():
    console = Console()

    # Stream the Python code with sliding window, then show full result
    stream_python_code_with_live(console, python_code, chunk_size=15, delay=0.01, window_lines=16)

if __name__ == "__main__":
    main()
