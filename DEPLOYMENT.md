# Deployment Guide

## Vercel Deployment

This application is configured for deployment on Vercel with the following setup:

### Prerequisites
- Node.js and npm (for Vercel CLI)
- Git repository pushed to GitHub/GitLab/Bitbucket

### Automatic Deployment

1. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign in with your GitHub account
   - Click "New Project" 
   - Select this repository

2. **Configuration:**
   - Vercel will automatically detect the `vercel.json` configuration
   - No additional environment variables needed for basic functionality
   - Build and deployment will start automatically

3. **Custom Domain (Optional):**
   - In Vercel dashboard, go to your project settings
   - Add custom domain under "Domains" tab

### Manual Deployment with Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from project root
vercel

# For production deployment
vercel --prod
```

### Configuration Files

- **`vercel.json`**: Main configuration for Vercel deployment
- **`api/app.py`**: Serverless function wrapper for Flask app
- **`runtime.txt`**: Specifies Python 3.9 runtime
- **`requirements.txt`**: Python dependencies
- **`.vercelignore`**: Files to exclude from deployment

### Limitations in Serverless Environment

1. **File Size**: Reduced max upload size to 100MB (from 500MB)
2. **Execution Time**: Vercel functions have 60-second timeout for Hobby plan
3. **Memory**: Limited to 1GB memory per function
4. **Storage**: Temporary storage only, files don't persist between requests

### Environment Variables

For production deployment, you may want to set:

```bash
# In Vercel dashboard under Settings > Environment Variables
FLASK_ENV=production
MAX_CONTENT_LENGTH=104857600  # 100MB in bytes
```

### Troubleshooting

**Common Issues:**

1. **Build Fails:**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version compatibility

2. **Function Timeout:**
   - Large tensor files may cause timeouts
   - Consider preprocessing files or splitting operations

3. **Memory Issues:**
   - Very large tensors may exceed memory limits
   - Implement streaming or chunked processing for large files

4. **Template Not Found:**
   - Ensure `templates/` directory is properly structured
   - Check that Flask template_folder path is correct

### Local Development

To test locally with Vercel development server:

```bash
# Install Vercel CLI
npm i -g vercel

# Start local development server
vercel dev
```

This will start a local server that mimics Vercel's serverless environment.

### Performance Optimization

For better performance in serverless environment:

1. **Optimize Dependencies:**
   - Use `torch` CPU version only
   - Consider lighter alternatives for numpy operations

2. **Lazy Loading:**
   - Import heavy libraries only when needed
   - Cache frequently used computations

3. **Request Optimization:**
   - Process smaller chunks of data
   - Implement pagination for large results

### Support

If you encounter deployment issues:

1. Check Vercel deployment logs in dashboard
2. Test locally with `vercel dev`
3. Refer to [Vercel Python documentation](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)