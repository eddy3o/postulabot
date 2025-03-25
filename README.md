# PostulaBot

PostulaBot is an intelligent assistant designed to help entrepreneurs apply for funding opportunities such as CORFO and other grants. It provides information on how to apply and offers assistance throughout the application process. PostulaBot is developed by Hello Future, a training company founded by Amanda Zerbinatti.

## Features

- **PDF Analysis**: Upload contest bases and get key insights using AI.
- **Dual Assistant**: Combines GPT-3.5 and Llama 3 for better responses.
- **Advanced RAG**: Semantic search in documents with Chroma DB.

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/eddy3o/postulabot.git
   cd postulabot
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```sh
   pip install -r requirements-mauricio.txt
   npm install
   ```

4. Set up environment variables:

   ```sh
   cp .env.example .env
   # Edit the .env file to include your OpenAI API key
   ```

5. Initialize the database:

   ```sh
   flask --app flaskr init-db
   ```

## Usage

1. Run the Flask application:

   ```sh
   flask --app flaskr run --debug
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000`.

## Project Structure

- `main.py`: The BOT
- `flaskr/`: Contains the Flask application modules.
  - `__init__.py`: Initializes the Flask app.
  - `auth.py`: Handles user authentication.
  - `bot.py`: Contains the bot routes and logic.
  - `db.py`: Manages database connections and initialization.
  - `static/`: Contains static files (CSS, images).
  - `templates/`: Contains HTML templates.
