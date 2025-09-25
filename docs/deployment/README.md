# Misfits! Deployment Guide

This guide covers different deployment scenarios for the Misfits! AI Life Simulation Game.

## Local Development Setup

### Prerequisites
- Python 3.10 or higher
- Git
- Optional: Local LLM service (Ollama, LM Studio, etc.)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-org/misfits-game.git
cd misfits-game

# Install dependencies
pip install -r requirements.txt

# Run setup script
python scripts/setup.py

# Start the game
python main.py
```

### With LLM Providers

#### Ollama Setup
```bash
# Install Ollama (see https://ollama.ai)
ollama pull llama2:7b-chat

# Verify it's running
curl http://localhost:11434/api/tags
```

#### LM Studio Setup
1. Download and install LM Studio
2. Load a compatible model (e.g., Llama 2 7B Chat)
3. Start the local server on port 1234

## Production Deployment

### Docker Deployment
```dockerfile
# Dockerfile (example)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python scripts/setup.py

EXPOSE 8000
CMD ["python", "main.py"]
```

### Server Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for game data and logs
- **Network**: Internet connection for LLM providers (if not using local)

### Configuration for Production

#### Environment Variables
```bash
export MISFITS_LLM_HOST=your-llm-host
export MISFITS_LLM_PORT=11434
export MISFITS_DB_PATH=/data/misfits.db
export MISFITS_DEBUG=false
```

#### Production Config
```yaml
# production_config.yaml
debug_mode: false
log_level: "INFO"
simulation:
  auto_save_interval: 300
database:
  path: "/data/misfits.db"
```

## Cloud Deployment

### AWS/GCP/Azure
1. Set up a virtual machine with adequate resources
2. Install Docker or Python environment
3. Configure persistent storage for game data
4. Set up monitoring and logging
5. Configure backups for save games

### Scaling Considerations
- The game is designed for single-instance deployment
- Multiple game instances can run independently
- Consider load balancing if running web interface

## Backup and Recovery

### Game Data Backup
```bash
# Backup game databases
cp misfits.db misfits_backup_$(date +%Y%m%d).db
cp memories.db memories_backup_$(date +%Y%m%d).db
cp vectors.db vectors_backup_$(date +%Y%m%d).db
```

### Configuration Backup
```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz data/
```

## Monitoring

### Health Checks
```bash
# Check if game is running
curl -f http://localhost:8000/health || exit 1

# Check LLM provider
python -c "
import asyncio
from src.utils.llm_interface import *
# Add health check code
"
```

### Log Monitoring
- Monitor `misfits.log` for errors
- Set up log rotation for production
- Consider centralized logging (ELK stack, etc.)

## Troubleshooting

### Common Issues

#### LLM Connection Problems
- Verify LLM service is running
- Check firewall settings
- Validate configuration

#### Database Issues
- Check file permissions
- Verify disk space
- Run database integrity checks

#### Performance Issues
- Monitor CPU and memory usage
- Check tick interval settings
- Consider reducing max_characters

### Debug Mode
```bash
python main.py --debug
```

### Getting Help
- Check the logs first
- Review configuration files
- Submit issues on GitHub with logs and config