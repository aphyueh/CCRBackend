# Color Cast Removal (CCR) Backend

A high-performance backend service for automatic color cast removal from images, powered by deep learning and deployed on Google Cloud Run for global accessibility with RESTful API endpoints.

## Overview

The Color Cast Removal backend is a production-ready API service that automatically detects and corrects color casts in images. Built with state-of-the-art deep learning models, Flask and Tensorflow, the service processes uplaoded images through a sophisticated color correction pipeline and returns the processed results for improved visual quality.

## Features

- **Automatic Color Cast Detection**: Advanced hybrid deep learning algorithms identify and quantify color casts
- **Real-time Processing**: Optimized inference pipeline for quick image processing
- **Scalable Architecture**: Containerized andDeployed on Google Cloud Run for automatic scaling
- **RESTful API**: Simple HTTP endpoints for easy integration
- **Multiple Input Formats**: Support for various image formats (JPEG, PNG, etc.)
- **Production Ready**: Robust error handling, CORS support and monitoring

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │────│  Flask API      │────│  TensorFlow     │
│                 │    │  (Cloud Run)    │    │  Inference      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │  Google Cloud   │
                       │   Storage       │
                       └─────────────────┘
```

## Core Technologies

### **Backend Framework**
- **Flask 3.1.0**: Lightweight, flexible Python web framework for building APIs
- **Flask-CORS 3.0.10**: Cross-Origin Resource Sharing support for web applications
- **Gunicorn 20.1.0**: Python WSGI HTTP Server for production deployment

### **Machine Learning & Computer Vision**
- **Tensorflow >= 2.8.0**: Open-source machine learning framework for model inference
- **OpenCV (opencv-contrib-python-headless>=4.5.0)**: Computer vision library for image processing
- **NumPy 1.26.4**: Numerical computing for efficient array operations
- **Pillow (PIL)**: Python Imaging Library for image 
- **scikit-image 0.25.2**: Image processing algorithms and utilities

### **Cloud Infrastructure**
- **Google Cloud Run**: Serverless container platform for deployment
- **Docker**: Containerization for consistent deployment environments
- **Google Cloud Storage 3.1.0**: Object storage for model artifacts and assets

### **Key Libraries**
- **Pydantic**: Data validation and settings management
- **Python-multipart**: File upload handling
- **Requests**: HTTP client library for external API calls

## Project Structure

```
CCRBackend/
├── main.py                 # FastAPI application entry point
├── model/
├── inference_pipeline.py   # Core inference logic and image processing
|   ├── model.py               # Neural network model definitions
|   ├── input.py               # Input validation and preprocessing
|   └── utils.py               # Utility functions and helpers
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── .dockerignore         # Docker ignore patterns
└── README.md             # Project documentation
```

## Core Components

### `main.py`
- Flask application setup and configuration
- CORS middleware and security settings
- Error handling and response formatting
- RESTful API endpoint definitions:
   - `/api/inference`: Primary color cast removal endpoint
   - `/api/adjust`: Manual image adjustment endpoint
   - `/api/histogram`: Image histogram analysis
   - `/api/init_model`: Model initialization endpoint
   - `/api/cleanup`: Maintenance endpoints

### `inference_pipeline.py`
- Implements color cast correction algorithms `remove_color_cast()` function using TensorFlow models
- Image format conversion and optimization
- Image preprocessing and postprocessing pipelines
- Leverages OpenCV, scikit-image, and NumPy for image transformations
- Optional advanced dehazing via OpenCV's ximgproc module
- Model inference orchestration
- Color cast correction algorithms

### `model.py`
- Neural network architecture definitions
- Tensorflow model loading and initialization
- Forward pass implementations
- Model configuration management
- Inference session handling

### `input.py`
- Request/response schemas using Pydantic
- Input validation and sanitization
- File upload handling
- Error response models

### `utils.py`
- Image utility functions
- Configuration management
- Logging setup
- Helper functions for common operations

### `Dockerfile`
- Based on Python 3.10 slim image
- Configures Gunicorn WSGI server on port 8080
- Sets up environment for Flask application

## Deployment

The service is deployed on **Google Cloud Run**, providing:

- **Automatic Scaling**: Scales from 0 to thousands of instances based on demand
- **Global Availability**: Deployed across multiple regions for low latency
- **Managed Infrastructure**: No server management required
- **Cost Optimization**: Pay-per-request model with automatic idle-down

### Deployment Architecture

1. **Container Build**: Docker image built with all dependencies
2. **Cloud Run Deployment**: Serverless deployment with automatic scaling
3. **Load Balancing**: Built-in load balancing for high availability
4. **Monitoring**: Integrated with Google Cloud Monitoring

## Installation & Setup

### Prerequisites
- Python 3.10+
- TensorFlow 2.8+
- Docker (for containerization)
- Google Cloud SDK (for deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/aphyueh/ccrbackend.git
   cd ccrbackend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**
   ```bash
   gunicorn --bind 0.0.0.0:8000 main:app
   ```

### Cloud Run Deployment

1. **Build and push to Container Registry**
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/[backend-name]
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy [backend-name] \
     --image gcr.io/[PROJECT-ID]/[backend-name] \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## API Endpoints
### Hello World Test
```
GET /api/hello
```
Simple test endpoint returning a `json` greeting message.

### Health Check
```
GET /health
```
Returns service health status and version information.

### Color Cast Removal
```
POST /api/inference
```
Processes images to automatically remove color casts.

**Request Format:**

- Content-Type: `multipart/form-data`
- Form Field: `image` (file upload)

**Response:**
- Processed image file as attachment

### Manual Image Adjustment
```
POST /api/adjust
```
Manually adjust image properties like brightness and temperature.

**Request Format:**

- Content-Type: multipart/form-data
- Form Field: image (file upload)
- Form Fields (optional):

   - brightness (float): Range typically -100 to 100
   - contrast (float): Range typically -100 to 100
   - saturation (float): Range typically -100 to 100
   - temperature (float): Range typically -100 to 100 (negative: cool, positive: warm)

**Response:** 
- Adjusted image in PNG format

### Histogram Analysis
```
POST /api/histogram
```
Generates RGB channel histograms for the uploaded image.

**Request Format:**

- Content-Type: `multipart/form-data`
- Form Field: image (file upload)

**Response Format:**
```json
json{
  "r": [0, 1, 5, ...],  // 256 values for red channel
  "g": [2, 3, 8, ...],  // 256 values for green channel
  "b": [1, 4, 7, ...]   // 256 values for blue channel
}
```
### Model Management
```
POST /api/init_model
```
Initializes the TensorFlow model for inference.

```
POST /api/cleanup
```
Cleans up temporary files created during processing.

## Technical Details

### Model Architecture
The color cast removal model uses a [deep neural network](https://github.com/k-cmy/color_cast_removal) specifically designed for color correction tasks:

- **Input**: RGB images of variable sizes
- **Architecture**: Custom CNN with attention mechanisms
- **Output**: Color-corrected RGB images
- **Training**: Trained on diverse datasets of color-cast and reference images

### Performance Optimizations
- **Batch Processing**: Efficient batch inference for multiple images
- **Memory Management**: Optimized memory usage for large images
- **Caching**: Model caching to reduce initialization overhead
- **Asynchronous Processing**: Non-blocking request handling

### Error Handling
- **Input Validation**: Comprehensive validation for uploaded files
- **Graceful Degradation**: Fallback mechanisms for edge cases
- **Detailed Logging**: Comprehensive logging for debugging
- **Error Responses**: Structured error messages with appropriate HTTP codes

## Monitoring & Observability

- **Health Checks**: Regular health check endpoints
- **Metrics Collection**: Custom metrics for processing time and success rates
- **Error Tracking**: Automatic error tracking and alerting
- **Performance Monitoring**: Real-time performance metrics

## Security

- **Input Sanitization**: Comprehensive input validation and sanitization
- **File Type Validation**: Strict validation of uploaded file types
- **Rate Limiting**: Built-in rate limiting for API endpoints
- **CORS Configuration**: Properly configured CORS policies

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or support:
- **GitHub Issues**: [Create an issue](https://github.com/aphyueh/CCRBackend/issues)
- **Documentation**: Check the inline code documentation
- **API Documentation**: Visit `/docs` endpoint when running the service

## Acknowledgments

- Built with modern Python ecosystem tools
- Deployed on Google Cloud Platform
- Inspired by latest research in computer vision and color science

---

**Note**: This backend service is part of a larger color cast removal application ecosystem. For the complete solution, check out the related frontend repositories https://github.com/aphyueh/CCRWebsite.

## Version History

- **v1.0.0**: Initial release with core color cast removal functionality
- **v1.1.0**: Performance optimizations and improved error handling
- **v1.2.0**: Enhanced model accuracy and additional preprocessing options

---

*Last updated: May 2025*