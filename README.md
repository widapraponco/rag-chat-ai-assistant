# Soravisi SQL Chat API - Powered by DeepSeek

A Flask-based REST API for natural language to SQL generation and execution using:

- **Vanna 2.0** - SQL AI framework
- **DeepSeek LLM** - AI model for SQL generation
- **BGE-M3 Embeddings** - For semantic search over table metadata
- **ChromaDB** - Vector database for storing embeddings
- **MySQL** - Target database for query execution

## Features

✅ **Natural Language to SQL** - Ask questions in plain English, get SQL queries
✅ **Automatic SQL Execution** - Execute generated queries against MySQL
✅ **DBT Integration** - Automatically loads table and column metadata from dbt catalog
✅ **Web Chat UI** - Modern, responsive chat interface
✅ **REST API** - Full-featured API for programmatic access
✅ **Vector Storage** - Semantic search over database schema using ChromaDB
✅ **DeepSeek LLM** - Powered by DeepSeek's advanced language model

## Project Structure

```
soravisi_api/
├── app/
│   ├── __init__.py           # Flask app factory
│   ├── routes.py             # API endpoints
│   ├── catalog_loader.py     # DBT catalog parser
│   ├── vanna_init.py         # Vanna initialization with DeepSeek
│   └── vanna_init.py
├── config/
│   └── settings.py           # Configuration management
├── templates/
│   └── chat.html             # Web chat interface
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Prerequisites

- Python 3.8+
- MySQL database (with access to sv_analytics)
- DeepSeek API Key (get from https://platform.deepseek.com/)
- DBT catalog.json from dbt project

## Installation

### 1. Clone / Setup Project

```bash
cd soravisi_api
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Flask
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000

# MySQL Database
MYSQL_HOST=45.77.35.173
MYSQL_USER=remote-soravisi
MYSQL_PASSWORD=your-actual-password
MYSQL_DATABASE=sv_analytics

# DeepSeek API (Required!)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# DBT Catalog Path (relative to soravisi_api folder)
DBT_CATALOG_PATH=../dbt_analytics/target/catalog.json
```

### 5. Prepare DBT Catalog

Make sure you have generated the DBT catalog:

```bash
cd ../dbt_analytics
.\venv\Scripts\Activate.ps1
dbt docs generate
```

This creates `target/catalog.json` with table and column metadata.

## Usage

### Start the API Server

```bash
python main.py
```

Server will be available at `http://localhost:5000`

### Web Chat Interface

Open your browser and go to: **http://localhost:5000/chat**

Ask questions like:

- "How many employees are in the system?"
- "Show me the top 10 users by join date"
- "What is the schema of the users table?"

### API Endpoints

#### Health Check

```bash
curl http://localhost:5000/health
```

#### Generate SQL (without execution)

```bash
curl -X POST http://localhost:5000/api/generate_sql \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users are there?"}'
```

Response:

```json
{
  "success": true,
  "question": "How many users are there?",
  "sql": "SELECT COUNT(*) as user_count FROM users;",
  "error": null
}
```

#### Execute SQL

```bash
curl -X POST http://localhost:5000/api/execute_sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM users LIMIT 5;"}'
```

Response:

```json
{
  "success": true,
  "sql": "SELECT * FROM users LIMIT 5;",
  "rows": [
    {"id": 1, "name": "John", "email": "john@example.com"},
    ...
  ],
  "row_count": 5,
  "error": null
}
```

#### Chat (Generate + Execute)

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many employees joined in 2024?",
    "execute": true
  }'
```

Response:

```json
{
  "success": true,
  "question": "How many employees joined in 2024?",
  "sql": "SELECT COUNT(*) FROM users WHERE YEAR(join_at) = 2024;",
  "results": [...],
  "executed": true,
  "error": null
}
```

#### List Tables

```bash
curl http://localhost:5000/api/tables
```

#### Get Column Info

```bash
curl http://localhost:5000/api/columns/users
```

## Configuration

### Vanna Settings (config/settings.py)

```python
# BGE-M3 embedding model
VANNA_EMBEDDING_MODEL = 'BAAI/bge-m3'

# ChromaDB storage path
CHROMADB_PATH = './data/chromadb'

# DeepSeek model
DEEPSEEK_MODEL = 'deepseek-chat'

# Max query results
MAX_QUERY_RESULTS = 1000
```

### Supported Models

#### Embedding Models

- **BAAI/bge-m3** (default) - State-of-the-art multilingual embeddings
- BAAI/bge-base-en-v1.5
- sentence-transformers/all-MiniLM-L6-v2

#### LLM Models (DeepSeek)

- **deepseek-chat** (default) - Best for SQL generation
- deepseek-coder - Better for code tasks

## Performance Tips

1. **Optimize ChromaDB**: Larger vector database improves context relevance
2. **Add Sample Queries**: More examples in knowledge base = better SQL generation
3. **Monitor API Usage**: Track DeepSeek API costs with logging
4. **Caching**: Consider caching common queries
5. **Connection Pooling**: MySQL connection pooling for production

## Troubleshooting

### DeepSeek API Errors

```
Error: Invalid API key
```

- Verify your DEEPSEEK_API_KEY in `.env`
- Check API key format (should start with `sk-`)
- Ensure account has API access at https://platform.deepseek.com/

### MySQL Connection Errors

```
Error: Access denied for user 'remote-soravisi'
```

- Verify MYSQL_PASSWORD is correct
- Check MySQL host/port connectivity
- Ensure user has SELECT/INSERT/CREATE permissions

### Missing ChromaDB

```
FileNotFoundError: No such file or directory: './data/chromadb'
```

- First run creates the directory automatically
- If issue persists: `mkdir -p data/chromadb`

### Embedding Model Download

First run downloads the BGE-M3 model (~600MB):

```
Downloading: 100%|██████████| 600M/600M [02:30<00:00, 4.0MB/s]
```

## Production Deployment

### Docker Deployment

```bash
docker build -t soravisi-api .
docker run -p 5000:5000 --env-file .env soravisi-api
```

### With Docker Compose

```bash
docker-compose up -d
```

### Environment Variables (Production)

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
export DEEPSEEK_API_KEY=sk-xxx
export MYSQL_PASSWORD=strong-password
python main.py
```

### Gunicorn (WSGI Server)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

## API Documentation

Full OpenAPI/Swagger documentation available at:

- JSON: `http://localhost:5000/api/swagger.json`
- UI: `http://localhost:5000/api/docs`

## Sample Queries

Common queries to test:

```sql
-- Count employees
SELECT COUNT(*) FROM users;

-- Users by department
SELECT department_id, COUNT(*) as count FROM users GROUP BY department_id;

-- Recent hires
SELECT name, email, join_at FROM users ORDER BY join_at DESC LIMIT 10;

-- Attendance stats
SELECT nama_pegawai, total_hadir FROM dim_attendance_performance;
```

## Integration with dbt

The API automatically loads from dbt catalog:

```yaml
# dbt_project.yml
name: "dbt_analytics"
version: "1.0.0"

models:
  dbt_analytics:
    staging:
      +materialized: view
    marts:
      +materialized: table
```

Generate catalog:

```bash
dbt docs generate
```

The catalog.json is automatically parsed and loaded into Vanna.

## Monitoring & Logging

Logs are output to console with timestamps:

```
2024-05-26 14:23:45,123 - app.routes - INFO - Initializing Vanna instance...
2024-05-26 14:23:47,456 - app.vanna_init - INFO - ✓ BGE-M3 embedding model loaded
2024-05-26 14:23:48,789 - app.vanna_init - INFO - ✓ DeepSeek LLM client initialized
```

For file logging, add to `config/settings.py`:

```python
logging.FileHandler('app.log')
```

## Roadmap

- [ ] User authentication & authorization
- [ ] Query history & analytics
- [ ] Custom prompt templates
- [ ] Multiple LLM providers (Claude, OpenAI, etc.)
- [ ] Query caching & optimization
- [ ] Web UI improvements (dark mode, export results)
- [ ] Real-time query monitoring

## Resources

- [Vanna Documentation](https://vanna.ai/docs)
- [DeepSeek API Docs](https://api-docs.deepseek.com/)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## Support

For issues:

1. Check logs in console output
2. Verify API keys in `.env`
3. Test MySQL connection: `mysql -h host -u user -p database`
4. Check model downloads: `ls data/chromadb/`

## License

Internal project for Soravisi Analytics
