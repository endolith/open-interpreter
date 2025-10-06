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
from rich.text import Text
from rich.align import Align


def stream_python_code_with_live(console, code_text, chunk_size=15, delay=0.05, window_fraction=0.4):
    """
    Stream Python code with a sliding window of the last N lines, then show full result.

    Args:
        console: Rich Console instance
        code_text: The Python code text to stream
        chunk_size: Number of characters per chunk (default: 15)
        delay: Delay between chunks in seconds (default: 0.05)
        window_fraction: Fraction of terminal height to use for sliding window (default: 0.4 = 40%)
    """
    accumulated_text = ""
    lines = []

    # Extract the actual Python code (remove the markdown code block markers)
    code_content = code_text.replace("```python\n", "").replace("```", "")

    # Calculate window size based on terminal height
    terminal_height = console.size.height
    window_lines = max(8, int(terminal_height * window_fraction))  # Minimum 8 lines

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
                # Create a single Text object with centered red ellipsis
                display_text = Text()
                # Add centered ellipsis by padding it
                terminal_width = console.size.width
                ellipsis_padding = (terminal_width - 3) // 2  # Center the 3-character "..."
                display_text.append(" " * ellipsis_padding + "...", style="red")
                display_text.append("\n")
                display_text.append('\n'.join(display_lines))
            else:
                display_text = Text('\n'.join(current_lines))

            # Update with styled text
            live.update(display_text)
            time.sleep(delay)

        # After streaming is complete, show the full syntax-highlighted version
        time.sleep(0.5)  # Brief pause before final display
        syntax = Syntax(code_content, "python", theme="ansi_dark", background_color="default")
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
                raise ValueError(f"Validation failed for data: {data}. The provided data does not meet the required schema specifications and validation criteria.")

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
            self.logger.info(f"Successfully processed batch {i//self.batch_size + 1} containing {len(batch)} records with comprehensive validation and transformation procedures")

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
        \"\"\"Transform a single record with comprehensive data processing and validation.\"\"\"
        transformed = record.copy()

        # Comprehensive data transformations with extensive validation and processing
        for key, value in transformed.items():
            if isinstance(value, str):
                # Apply comprehensive string transformations including normalization, cleaning, and formatting
                transformed[key] = value.strip().lower().replace('  ', ' ').replace('\t', ' ')
            elif isinstance(value, (int, float)):
                # Apply numerical transformations with precision control and validation
                transformed[key] = round(value, 2) if isinstance(value, float) else value
            elif isinstance(value, dict):
                # Handle nested dictionary structures with recursive transformation
                transformed[key] = {k: v.strip().lower() if isinstance(v, str) else v for k, v in value.items()}

        return transformed

    def get_metrics_summary(self) -> Dict[str, Any]:
        \"\"\"Get processing metrics summary.\"\"\"
        return {
            'duration': self.metrics.duration,
            'records_processed': self.metrics.records_processed,
            'errors_count': self.metrics.errors_count,
            'success_rate': (self.metrics.records_processed - self.metrics.errors_count) / max(self.metrics.records_processed, 1) * 100,
            'average_processing_time_per_record': self.metrics.duration / max(self.metrics.records_processed, 1),
            'memory_efficiency_score': (self.metrics.records_processed - self.metrics.errors_count) / max(self.metrics.records_processed, 1) * 100,
            'throughput_records_per_second': self.metrics.records_processed / max(self.metrics.duration, 0.001)
        }


# Example usage
async def main():
    \"\"\"Example usage of the DataProcessor.\"\"\"
    # Create processor
    processor = DataProcessor(batch_size=50)

    # Add validator
    validator = SchemaValidator(['id', 'name', 'value'])
    processor.add_validator(validator)

    # Sample data with comprehensive test cases covering various data validation scenarios and edge cases
    sample_data = [
        {'id': 1, 'name': 'Item 1', 'value': 10.5, 'description': 'This is a comprehensive test item with extensive metadata and validation requirements'},
        {'id': 2, 'name': 'Item 2', 'value': 20.3, 'description': 'Another test item designed to validate complex data processing pipelines and error handling mechanisms'},
        {'id': 3, 'name': 'Item 3', 'value': 15.7, 'description': 'Final test item to ensure complete coverage of all validation scenarios and transformation procedures'},
        {'id': 4, 'name': 'Item 4', 'value': 25.9, 'description': 'Additional test case for comprehensive validation of data processing algorithms and performance metrics'},
        {'id': 5, 'name': 'Item 5', 'value': 30.1, 'description': 'Extended test case to validate complex data transformation pipelines and error recovery mechanisms'}
    ]

    # Process data with comprehensive error handling and performance monitoring
    try:
        result = await processor.process_data_async(sample_data)
        print("Processing completed successfully! All data has been validated and transformed according to the specified schema requirements and comprehensive validation criteria.")
        print(f"Comprehensive metrics summary with detailed performance analysis: {processor.get_metrics_summary()}")
        print("Data processing pipeline executed with optimal performance, comprehensive error handling mechanisms, and advanced monitoring capabilities.")
        print("All validation rules have been successfully applied, data transformations completed, and performance metrics recorded for analysis.")
    except Exception as e:
        print(f"Processing failed with comprehensive error details: {e}. Please check your data format, validation rules, and ensure all required fields are present.")
        print("Error occurred during data processing pipeline execution. Please review the data structure and validation requirements.")


if __name__ == "__main__":
    asyncio.run(main())
```"""


def main():
    console = Console()

    # Stream the Python code with sliding window, then show full result
    stream_python_code_with_live(console, python_code, chunk_size=15, delay=0.1, window_fraction=0.75)

if __name__ == "__main__":
    main()
