🔄 Claude AI Batch Translator
A comprehensive Streamlit app for cost-effective document translation using Claude's Batch API. Save up to 50% on translation costs while processing large document collections with intelligent chunking and automatic file reassembly.
Show Image
✨ Features

💰 Cost Savings: Up to 50% lower costs compared to real-time API
🔄 Batch Processing: Handle large document collections efficiently
🧠 Smart Chunking: Automatically split large files with intelligent overlap
📊 Job Monitoring: Real-time status tracking of translation jobs
🔗 File Reassembly: Smart reconstruction of chunked translations
🌐 Multi-Language: Support for all Claude-supported language pairs
📁 Bulk Upload: Process multiple files simultaneously
⚙️ Custom Instructions: Add domain-specific translation guidelines

🚀 How to Use

Configure - Set up your Claude API key, languages, and processing parameters
Submit Batch - Upload files, review costs, and submit translation jobs
Monitor Jobs - Track progress of your batch translations (up to 24 hours)
Download Results - Retrieve and save your translated documents

🏃 How to run it on your own machine

Clone the repository
bashgit clone <your-repo-url>
cd BatchTranslationCustomInstruction

Install the requirements
bashpip install -r requirements.txt

Run the app
bashstreamlit run streamlit_app.py


📋 Requirements
Create a requirements.txt file with the following dependencies:
streamlit>=1.28.0
anthropic>=0.8.0
aiofiles>=23.0.0
tiktoken>=0.5.0
nest-asyncio>=1.5.0
pandas>=2.0.0
⚙️ Configuration
Before using the app, you'll need:

Claude API Key - Get one from Anthropic Console
Batch API Access - Ensure your account has batch processing enabled
Input Files - Documents in supported formats (txt, md, etc.)

💡 Best For

Large Document Sets (50+ files)
Cost-Sensitive Projects (budget-conscious translations)
Non-Urgent Work (24-hour processing window)
Academic Research (bulk paper translations)
Business Documentation (manual and report translations)

🔧 Advanced Features

Custom Translation Instructions - Add domain-specific guidelines
Multiple Model Support - Choose from Claude 3 and Claude 4 models
Job Persistence - Resume monitoring after interruptions
Automatic Cleanup - Temporary file management
Cost Estimation - Preview costs before submission

📊 Batch vs Real-time Comparison
FeatureBatch APIReal-time APICost50% lowerStandard pricingProcessing TimeUp to 24 hoursImmediateScaleUnlimited filesRate limitedBest ForLarge volumesSmall/urgent jobs
🛡️ Security

API keys are handled securely through Streamlit's session state
Temporary files are automatically cleaned up
No data is stored permanently on the server

📝 License
This project is open source and available under the MIT License.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📞 Support
If you encounter any issues or have questions:

Check the app's built-in help and status messages
Review Claude's Batch API documentation
Open an issue in this repository


Built with ❤️ using Streamlit and Anthropic Claude