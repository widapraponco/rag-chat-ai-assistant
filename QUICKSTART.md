# Quick Start Guide - Soravisi API

## 🚀 5-Minute Setup

### Step 1: Navigate to project

```powershell
cd c:\Users\widap\Documents\Python\soravisiAI\soravisi_api
```

### Step 2: Create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

This installs:

- Flask + Flask-CORS for API
- Vanna 2.0 for SQL generation
- DeepSeek API client
- ChromaDB for vector storage
- BGE-M3 embedding model
- MySQL connector

### Step 4: Setup environment variables

```powershell
Copy-Item .env.example .env
# Edit .env with your credentials
notepad .env
```

Required settings:

- `DEEPSEEK_API_KEY` - Get from https://platform.deepseek.com/
- `MYSQL_PASSWORD` - Your MySQL password

### Step 5: Start the API

```powershell
python main.py
```

### Step 6: Open chat interface

Visit: **http://localhost:5000/chat**

---

## 📝 Configuration

### MySQL Connection

```
Host: 45.77.35.173
User: remote-soravisi
Database: sv_analytics
Port: 3306
```

### DeepSeek API

- Get API key from: https://platform.deepseek.com/
- Set in .env: `DEEPSEEK_API_KEY=sk-xxx...`
- Model: deepseek-chat

### DBT Catalog

The API automatically loads table metadata from:

```
../dbt_analytics/target/catalog.json
```

Make sure to run `dbt docs generate` in the dbt project first.

---

## 🧪 Testing

### Test API Health

```bash
curl http://localhost:5000/health
```

### Test SQL Generation

```bash
curl -X POST http://localhost:5000/api/generate_sql `
  -H "Content-Type: application/json" `
  -d '{"question":"How many users?"}'
```

### Test with Chat UI

Open browser: http://localhost:5000/chat
Ask: "How many employees are there?"

---

## 📚 API Endpoints

| Method | Endpoint               | Purpose                    |
| ------ | ---------------------- | -------------------------- |
| GET    | `/health`              | Health check               |
| GET    | `/`                    | API info                   |
| GET    | `/chat`                | Web chat UI                |
| POST   | `/api/generate_sql`    | Generate SQL from question |
| POST   | `/api/execute_sql`     | Execute SQL query          |
| POST   | `/api/chat`            | Generate + Execute         |
| GET    | `/api/tables`          | List all tables            |
| GET    | `/api/columns/<table>` | Get table columns          |

---

## 🔧 Troubleshooting

### Issue: "No module named 'vanna'"

**Solution:** Make sure you activated the venv and installed requirements

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "DeepSeek API key invalid"

**Solution:**

1. Get key from https://platform.deepseek.com/api_keys
2. Check format: should start with `sk-`
3. Update .env file with correct key
4. Restart the app

### Issue: "Access denied for MySQL user"

**Solution:** Verify credentials in .env

```powershell
# Test connection
mysql -h 45.77.35.173 -u remote-soravisi -p -D sv_analytics
```

### Issue: "Catalog.json not found"

**Solution:** Generate it from dbt project

```powershell
cd ../dbt_analytics
.\venv\Scripts\Activate.ps1
dbt docs generate
```

---

## 🚢 Deployment

### With Docker

```powershell
docker build -t soravisi-api .
docker run -p 5000:5000 --env-file .env soravisi-api
```

### With Docker Compose

```powershell
docker-compose up -d
```

### Production (Gunicorn)

```powershell
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

---

## 📊 Architecture

```
User Query
    ↓
[Flask API] → [DeepSeek LLM] → SQL Generation
    ↓
[BGE-M3 Embeddings] ← [ChromaDB] ← [dbt Catalog]
    ↓
[MySQL Database] ← SQL Execution
    ↓
Results
```

**Components:**

- **Flask**: REST API framework
- **Vanna 2.0**: SQL AI framework
- **DeepSeek**: Language model for SQL
- **BGE-M3**: Multilingual embeddings for semantic search
- **ChromaDB**: Vector database for metadata
- **MySQL**: Target database

---

## 💡 Usage Examples

### Example 1: Simple Count

**Question:** "How many users in the system?"
**Generated SQL:** `SELECT COUNT(*) FROM users;`

### Example 2: With Filters

**Question:** "Show me employees who joined in 2024"
**Generated SQL:** `SELECT * FROM users WHERE YEAR(join_at) = 2024;`

### Example 3: Aggregation

**Question:** "What's the average salary by department?"
**Generated SQL:** `SELECT department_id, AVG(basesalary) FROM users GROUP BY department_id;`

---

## 📞 Support

For issues:

1. Check `.env` configuration
2. Verify DeepSeek API key is valid
3. Test MySQL connection
4. Check logs for error messages
5. Review README.md for more details

---

Last Updated: 2024-05-26
