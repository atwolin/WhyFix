# Streamlit Module

This module provides interactive web applications for the WhyFix L2 learner error correction system. It offers user-friendly interfaces for real-time error explanation, data visualization, and research result analysis.

## Overview

The Streamlit module delivers comprehensive web-based interfaces for:

1. **WhyFix Interactive System**: Real-time sentence error explanation and correction
2. **Research Data Viewer**: Interactive exploration of experimental results
3. **Method Comparison Interface**: Side-by-side analysis of different approaches
4. **Educational Interface**: User-friendly tools for language learners and teachers

## Directory Structure

```
streamlit/
├── README.md                   # This file
├── requirements.txt           # Application dependencies
└── src/                      # Source code and applications
    ├── main.py               # Main application entry point
    ├── components/           # Reusable UI components
    ├── pages/               # Individual application pages
    ├── utils/               # Utility functions
    └── config/              # Configuration files
```

## Key Applications

### WhyFix Interactive System

Real-time error correction and explanation interface:

**Core Features:**
- **Dual Input Interface**: Original and corrected sentence input
- **Real-time Processing**: Immediate error detection and explanation
- **Multi-method Analysis**: Comparison across different explanation approaches
- **ERRANT Integration**: Automatic error annotation and classification
- **Educational Feedback**: Clear, pedagogically sound explanations

**User Workflow:**
1. **Input**: Enter learner sentence and corrected version
2. **Processing**: Automatic error detection and classification
3. **Analysis**: Multi-method explanation generation
4. **Display**: Clear, comparative result presentation
5. **Learning**: Educational insights and improvement suggestions

### Research Data Viewer

Comprehensive interface for experimental result analysis:

**Analytical Features:**
- **Dataset Navigation**: Browse FCE and Longman datasets
- **Method Filtering**: Compare baseline, RAG, and hybrid approaches
- **Performance Metrics**: Interactive display of evaluation results
- **Statistical Analysis**: Significance testing and confidence intervals
- **Visualization**: Charts and graphs for result interpretation

**Research Capabilities:**
- **Experiment Comparison**: Side-by-side method evaluation
- **Parameter Analysis**: Impact of different configuration settings
- **Quality Assessment**: Error analysis and system performance
- **Publication Ready**: Export capabilities for academic use

### Method Comparison Interface

Detailed comparison system for different explanation approaches:

**Comparison Features:**
- **Method-by-Method**: Direct comparison of explanation quality
- **Parameter Sensitivity**: Impact of different settings
- **Performance Benchmarks**: Speed and accuracy metrics
- **User Experience**: Subjective quality assessment tools

## Technical Implementation

### Application Architecture

**Frontend Components:**
- **Streamlit Framework**: Modern, responsive web interface
- **Interactive Widgets**: Dynamic form elements and controls
- **Real-time Updates**: Live data processing and display
- **Responsive Design**: Mobile and desktop compatibility

**Backend Integration:**
- **Model APIs**: Direct integration with explanation models
- **Data Processing**: Real-time preprocessing and formatting
- **Result Caching**: Efficient response time optimization
- **Error Handling**: Robust error management and user feedback

### User Interface Design

**Design Principles:**
- **Simplicity**: Clean, intuitive interface design
- **Accessibility**: Support for diverse user needs
- **Performance**: Fast loading and responsive interactions
- **Educational Focus**: Clear, pedagogically appropriate presentation

**Interface Elements:**
- **Input Forms**: User-friendly data entry
- **Result Display**: Clear, organized output presentation
- **Navigation**: Intuitive page and section navigation
- **Help System**: Integrated guidance and documentation

## Usage Examples

### Running the Application

```bash
# Navigate to streamlit directory
cd /path/to/WhyFix/streamlit

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run src/main.py
```

### Configuration

**Environment Setup:**
```bash
# Set required environment variables
export OPENAI_API_KEY="your_api_key_here"
export STREAMLIT_SERVER_PORT=8501
```

**Application Configuration:**
```python
# streamlit config
st.set_page_config(
    page_title="WhyFix - L2 Error Correction",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Integration with WhyFix System

**Data Flow:**
```
User Input → Streamlit Interface → WhyFix Backend → Model Processing → Result Display
```

**API Integration:**
- **Real-time Processing**: Direct model API calls
- **Batch Processing**: Integration with batch result systems
- **Data Management**: Connection to preprocessed datasets
- **Quality Assurance**: Validation of processing results

## Features in Detail

### Real-time Error Explanation

**Process Flow:**
1. **Input Validation**: Check sentence format and content
2. **Error Detection**: ERRANT-based automatic annotation
3. **Context Enhancement**: RAG system integration
4. **Explanation Generation**: Multi-method processing
5. **Result Presentation**: Comparative display of explanations

**User Experience:**
- **Immediate Feedback**: Sub-second response times
- **Clear Explanations**: Pedagogically appropriate language
- **Visual Indicators**: Error highlighting and categorization
- **Learning Support**: Educational context and improvement suggestions

### Data Exploration Interface

**Research Tools:**
- **Dataset Browser**: Navigate through processed datasets
- **Filter System**: Narrow results by method, dataset, or parameters
- **Visualization Tools**: Charts, graphs, and statistical displays
- **Export Functions**: Download results for further analysis

### Performance Monitoring

**System Metrics:**
- **Response Time**: Real-time performance monitoring
- **User Engagement**: Usage statistics and patterns
- **System Health**: Error rates and performance indicators
- **Quality Metrics**: User satisfaction and accuracy measures

## Integration Points

### WhyFix System Integration

**Module Connections:**
- **Preprocess**: Access to cleaned and formatted datasets
- **Process**: Integration with RAG and retrieval systems
- **Models**: Direct API access to explanation generation
- **Postprocess**: Display of evaluation metrics and analysis

**Data Sources:**
- **Live Processing**: Real-time model API calls
- **Cached Results**: Pre-computed experimental results
- **Static Data**: Reference datasets and knowledge bases

## Deployment and Configuration

### Local Development

**Setup Requirements:**
- Python 3.8+ environment
- Streamlit framework
- WhyFix system dependencies
- API access credentials

**Development Workflow:**
1. **Environment Setup**: Install dependencies and configure APIs
2. **Local Testing**: Run application in development mode
3. **Feature Development**: Implement and test new capabilities
4. **Integration Testing**: Validate with WhyFix backend systems

### Production Deployment

**Deployment Options:**
- **Local Server**: Single-user or small team deployment
- **Cloud Hosting**: Scalable web application hosting
- **Containerization**: Docker-based deployment for consistency
- **Load Balancing**: Multi-instance deployment for high availability

**Configuration Management:**
- **Environment Variables**: Secure API key and configuration management
- **Resource Allocation**: Memory and CPU optimization
- **Security Settings**: User authentication and data protection
- **Monitoring**: Application performance and error tracking

## Dependencies

### Core Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
openai>=1.0.0
```

### WhyFix Integration

```
errant>=2.3.3
sentence-transformers>=2.2.0
spacy>=3.6.0
nltk>=3.8.0
```

### Optional Enhancements

```
streamlit-components-template
streamlit-plotly-events
streamlit-aggrid
streamlit-option-menu
```

This module serves as the primary user interface for the WhyFix system, making advanced L2 learner error correction research accessible through intuitive, interactive web applications for educators, researchers, and language learners.