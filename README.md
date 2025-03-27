# Fashion Agent

## Running the Fashion Agent App

### GitHub Repository

Clone the repository from GitHub:

```sh
git clone <repository-url>
cd <repository-folder>
```

### Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

### Running the Application

Run the Streamlit app with:

```sh
streamlit run fashion_agent.py
```

## Setting Up Environment Variables

To run this app, you need to configure the required environment variables.

### Step 1: Create the .env File

1. Copy the `.env.example` file and rename it to `.env`.
2. Open the `.env` file and update the values with your own credentials and configurations.

### Step 2: Required Environment Variables

Ensure the following variables are correctly set:

#### OpenAI API Configuration

```sh
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-35-turbo-16k"
AZURE_OPENAI_API_VERSION="your-api-version"
AZURE_OPENAI_DEPLOYMENT_NAME_4o="gpt-4o"
```

#### LangChain Configuration

```sh
LANGCHAIN_API_KEY="your-langchain-api-key"
LANGCHAIN_TRACING_V2=true
```

#### Tavily API Key

```sh
TAVILY_API_KEY="your-tavily-api-key"
```

#### Database Configuration

```sh
DATABASE_URL="your-database-connection-url"
```

### Step 3: Verify Environment Variables

Before running the app, ensure all required variables are correctly filled in the `.env` file.

