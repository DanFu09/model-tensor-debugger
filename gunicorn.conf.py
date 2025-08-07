# Gunicorn configuration for production deployment
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes - optimized for Render
workers = 1  # Single worker to avoid memory issues on limited RAM
worker_class = "sync"
worker_connections = 1000
timeout = 600  # Longer timeout for large file processing
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 100  # More frequent restarts to prevent memory leaks
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'model-tensor-debugger'

# Server mechanics
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment and configure for HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Worker timeout and memory limits
worker_tmp_dir = None
preload_app = True

# For large file uploads (tensor files can be big)
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190
max_request_size = 536870912  # 512MB