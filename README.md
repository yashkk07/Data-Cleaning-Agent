# Data Cleaning Agent

An intelligent ETL pipeline powered by AI that automatically detects and corrects data quality issues in CSV files. The agent uses LLM-based planning to generate and execute data cleaning transformations.

## Overview

This application provides an automated data cleaning solution with the following features:

- **Intelligent Data Profiling**: Analyzes CSV files to detect issues like missing values, data type inconsistencies, and duplicates
- **AI-Powered Planning**: Uses LLM (OpenAI/Groq) to generate optimal cleaning strategies
- **Automated Execution**: Applies transformations such as:
  - Column name standardization
  - Missing value handling
  - Whitespace trimming
  - Duplicate removal
  - Type conversion and parsing
  - Date/time parsing
  - Currency and percentage normalization
  - Boolean parsing
  - Outlier detection
- **Validation & Assessment**: Validates transformations and provides confidence metrics
- **Web Interface**: Flask-based UI for uploading files and downloading cleaned results

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Data-Cleaning-Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create `.env` file):
   ```
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   FLASK_SECRET=your_flask_secret
   ```

## Usage

### Running the Web Application

```bash
python app.py
```

Then navigate to `http://localhost:5000` in your browser.

1. Upload a CSV file
2. Select cleaning mode (full pipeline or specific cleaners)
3. Download the cleaned CSV

### Running Tests

```bash
python -m pytest tests/
```

## Project Structure

```
Data-Cleaning-Agent/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── etl/
│   ├── pipeline.py       # Main ETL pipeline orchestration
│   ├── agent/            # Agent loop logic
│   ├── assessment/        # Confidence and readiness assessment
│   ├── executor/         # Code execution and safety
│   ├── extract/          # CSV/JSON reading utilities
│   ├── llm/              # LLM planning and integration
│   ├── load/             # Output writing
│   ├── profile/          # Data profiling and serialization
│   ├── transform/        # Data cleaning transformations
│   └── validate/         # Validation and feedback
├── data/
│   ├── uploads/          # User uploaded CSV files
│   ├── outputs/          # Cleaned CSV files
│   └── test_files/       # Test data
├── templates/            # HTML templates for web UI
├── tests/                # Unit and integration tests
└── logs/                 # Application logs
```

## Architecture

The ETL pipeline follows these steps:

1. **Extract**: Read and parse input CSV files with automatic encoding detection
2. **Profile**: Generate statistical profiles and identify data quality issues
3. **Plan**: Use LLM to generate a cleaning plan based on profiles
4. **Execute**: Apply transformations in an iterative loop with feedback
5. **Validate**: Validate transformations and collect results
6. **Assess**: Compute confidence scores and readiness assessment
7. **Load**: Write cleaned data to output CSV

## Technologies

- **Python 3.8+**: Core language
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **OpenAI/Groq**: LLM integration
- **Chardet**: Character encoding detection

## Configuration

Key configuration files:
- `.env`: Environment variables (API keys, secrets)
- `app.py`: Flask app configuration
- `etl/pipeline.py`: Pipeline parameters

## Contributing

Contributions are welcome! Please follow these steps:

1. Create a feature branch
2. Make your changes
3. Add/update tests
4. Submit a pull request

## License

MIT License

