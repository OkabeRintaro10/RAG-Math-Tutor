# Project Overview

This project is a Math Tutor application that uses a Retrieval-Augmented Generation (RAG) pipeline to answer math-related questions. It consists of a Python backend and a React frontend.

**Backend:**

*   **Framework:** FastAPI
*   **Asynchronous Task Processing:** Inngest
*   **LLM Orchestration:** LangChain and LangGraph
*   **Vector Store:** Qdrant
*   **Dependencies:** `google-genai`, `qdrant-client`, `langchain`, `fastapi`, `inngest`, etc.

The backend exposes a REST API for ingesting PDF documents into the knowledge base and for answering questions using the RAG pipeline. It uses Guardrails to ensure that the questions are math-related and that the answers are appropriate.

**Frontend:**

*   **Framework:** React
*   **Build Tool:** Vite
*   **Dependencies:** `axios`, `react-latex-next`, `katex`

The frontend provides a chat interface for users to interact with the Math Tutor. It supports rendering LaTeX for mathematical equations and allows users to provide feedback on the answers.

# Building and Running

## Backend

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## Frontend

1.  **Navigate to the UI directory:**
    ```bash
    cd ui/math-tutor-ui
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Run the development server:**
    ```bash
    npm run dev
    ```

# Development Conventions

*   The backend configuration is managed through `config/config.yaml` and `params.yaml`.
*   The backend uses a structured logging format.
*   The frontend code is located in the `ui/math-tutor-ui/src` directory.
*   The main React component is `App.jsx`.
