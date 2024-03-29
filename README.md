
# Chat with PDF using AWS Bedrock 💁

This is an interactive chat application powered by AWS Bedrock. It allows users to ask questions related to PDF files and get responses generated by AI models.

## Installation

1. Clone the repository:


2. Navigate to the project directory:


3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## AWS Credentials Setup

To run this application, you need to set up your AWS credentials. You can do this by creating a `.env` file in the project directory and adding your AWS access key ID, AWS secret access key, and AWS default region in the following format:

```toml
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
AWS_DEFAULT_REGION=<your-default-region>
```

## Data File to Download

You can download the PDF file from the following link:
[Download PDF](https://www.researchgate.net/publication/370653602_Generative_AI)

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. You can ask questions related to the PDF file in the text input field.

3. Use the buttons to perform various actions such as updating vector stores or getting responses from different AI models.

## Features

- Update or create vector stores for better response generation.
- Choose from different AI models (e.g., Claude, Llama2) to get responses.
- Easy-to-use interface powered by Streamlit.
