# Testing Vercel Deployment Locally

## Method 1: Vercel Dev Server (Most Accurate)

### 1. Install Vercel CLI
```bash
npm install -g vercel
```

### 2. Login to Vercel (Optional but Recommended)
```bash
vercel login
```

### 3. Start Local Development Server
```bash
# Navigate to project root
cd /path/to/model-tensor-debugger

# Start Vercel development server
vercel dev
```

This will:
- Start a local server that mimics Vercel's serverless functions
- Handle routing exactly like production
- Show you any deployment issues before you deploy
- Typically runs on http://localhost:3000

### 4. Test the Application
- Visit http://localhost:3000 (or whatever port Vercel shows)
- Test file uploads, tensor comparisons, all features
- Check browser console for any errors

## Method 2: Direct Function Testing

Test the serverless function directly:

```bash
# Install dependencies in conda environment
source /Users/danfu/anaconda3/bin/activate ml-debug-viz

# Test the API wrapper directly
cd api
python -c "from app import handler; print('Handler loaded successfully')"
```

## Method 3: Verification Script

Run our custom verification script:

```bash
# In project root
python verify_deployment.py
```

## Method 4: Docker Simulation (Advanced)

Create a Dockerfile to simulate serverless environment:

```dockerfile
FROM python:3.9-slim

WORKDIR /var/task

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Set environment
ENV PYTHONPATH=/var/task

# Start with serverless function
CMD ["python", "api/app.py"]
```

## Common Issues When Testing Locally

### Issue 1: Import Path Problems
**Symptom**: `ModuleNotFoundError` when running `vercel dev`
**Fix**: 
```bash
# Make sure you're in the project root
export PYTHONPATH=$(pwd)
vercel dev
```

### Issue 2: Template Not Found
**Symptom**: `TemplateNotFound: index.html`
**Fix**: Check that the template path is correct in `app.py`:
```python
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
```

### Issue 3: Large File Upload Failures
**Symptom**: Uploads fail or timeout
**Fix**: Test with smaller files first (< 10MB) to ensure basic functionality works

### Issue 4: Memory Issues
**Symptom**: Process killed or out of memory errors
**Fix**: Vercel has memory limits - test with smaller tensor files

## Debug Mode

For more verbose output during local testing:

```bash
# Set debug environment
export FLASK_DEBUG=1
export FLASK_ENV=development

# Run vercel dev with debug output
vercel dev --debug
```

## Production-like Testing

To test exactly like production:

```bash
# Build for production
vercel build

# Test the built version
vercel dev --prod
```

## Testing Checklist

When testing locally, verify:

- [ ] Main page loads at http://localhost:3000
- [ ] Health endpoint works: http://localhost:3000/health  
- [ ] File upload interface appears
- [ ] Archive uploads work (test with small files)
- [ ] .pth file uploads work
- [ ] Tensor comparison displays
- [ ] "Inspect Raw Tensor Values" works
- [ ] No JavaScript console errors
- [ ] No Python errors in terminal

## Performance Testing

Test serverless limitations:

```bash
# Create a test script
cat > test_limits.py << EOF
import requests
import time

# Test response time
start = time.time()
response = requests.get('http://localhost:3000/health')
print(f"Health check: {time.time() - start:.2f}s")

# Test with file upload (use small test file)
files = {'model1': open('test_file1.pth', 'rb')}
start = time.time()
response = requests.post('http://localhost:3000/upload', files=files)
print(f"Upload: {time.time() - start:.2f}s")
EOF

python test_limits.py
```

## Next Steps After Local Testing

Once local testing passes:

1. **Deploy to preview**: `vercel` (creates preview deployment)
2. **Test preview URL**: Verify everything works in real Vercel environment
3. **Deploy to production**: `vercel --prod`

## Troubleshooting Commands

```bash
# Check Vercel logs
vercel logs

# Inspect build output
vercel inspect

# Remove local Vercel cache
rm -rf .vercel

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install -g vercel
```